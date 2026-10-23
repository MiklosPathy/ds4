/* V4.1 Q8 production-shape harness with an independent sparse double oracle,
 * cast-sensitive fixtures and full-output parity against unchanged scalar
 * production calls. Explicit copies, guards and synchronization after stages. */
#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)
#define RUN(x) do { fprintf(stderr, "stage: %s\n", #x); CHECK(x); CHECK(sync_guards()); } while (0)

enum { GUARD_BYTES = 64 };
typedef struct allocation {
    ds4_gpu_tensor *storage, *view;
    size_t bytes;
    struct allocation *next;
} allocation;
static allocation *allocations;

static int sync_guards(void) {
    CHECK(ds4_gpu_synchronize());
    for (allocation *a = allocations; a; a = a->next) {
        unsigned char before[GUARD_BYTES], after[GUARD_BYTES];
        CHECK(ds4_gpu_tensor_read(a->storage, 0, before, sizeof(before)));
        CHECK(ds4_gpu_tensor_read(a->storage, GUARD_BYTES + a->bytes, after, sizeof(after)));
        for (unsigned i = 0; i < GUARD_BYTES; i++) CHECK(before[i] == 0xa5 && after[i] == 0xa5);
    }
    return 1;
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    allocation *a = calloc(1, sizeof(*a));
    if (!a || bytes > SIZE_MAX - 2 * GUARD_BYTES) { free(a); return NULL; }
    a->bytes = bytes;
    a->storage = ds4_gpu_tensor_alloc(bytes + 2 * GUARD_BYTES);
    if (!a->storage) { free(a); return NULL; }
    unsigned char guard[GUARD_BYTES];
    memset(guard, 0xa5, sizeof(guard));
    if (!ds4_gpu_tensor_write(a->storage, 0, guard, sizeof(guard)) ||
        !ds4_gpu_tensor_write(a->storage, GUARD_BYTES + bytes, guard, sizeof(guard))) goto fail;
    a->view = ds4_gpu_tensor_view(a->storage, GUARD_BYTES, bytes);
    if (!a->view || (data && !ds4_gpu_tensor_write(a->view, 0, data, bytes))) goto fail;
    a->next = allocations;
    allocations = a;
    return a->view;
fail:
    ds4_gpu_tensor_free(a->view);
    ds4_gpu_tensor_free(a->storage);
    free(a);
    return NULL;
}

static void guarded_free(ds4_gpu_tensor *t) {
    for (allocation **link = &allocations; *link; link = &(*link)->next) {
        allocation *a = *link;
        if (a->view == t) {
            *link = a->next;
            ds4_gpu_tensor_free(a->view);
            ds4_gpu_tensor_free(a->storage);
            free(a);
            return;
        }
    }
    ds4_gpu_tensor_free(t);
}
#define ds4_gpu_tensor_free guarded_free

#include <sys/mman.h>


/* Independent sparse MXFP4 oracle. Encoded 0x22 lanes are +1 times the
 * E8M0 scale; 0xaa lanes are -1. All other blocks encode exact zero. */
enum { E=384, U=6, K=5120, H=2304, M=5120 };
typedef struct { uint8_t e, qs[16]; } mx;
static void route(int32_t *ids,unsigned rank,unsigned mode) {
    const int32_t mixed[6]={0,191,192,193,382,383};
    for(unsigned j=0;j<U;j++)
        ids[j]=mode==2?mixed[j]:(int32_t)((mode?1-rank:rank)*192+j);
}
static float gate_ref(unsigned e,unsigned row,unsigned tok,int up) {
    if(row%127)return 0.f;
    unsigned block=(e+row)%160;
    float x=(tok+block)&1?127.f:-127.f;
    return x*32.f*(up?-1.f:1.f)/2048.f;
}
static int run(unsigned n,unsigned rank,unsigned mode) {
    const size_t ge=(size_t)H*(K/32)*sizeof(mx),de=(size_t)M*(H/32)*sizeof(mx);
    const size_t table=E*ge,wb=2*table+E*de;
    unsigned char *model=mmap(NULL,wb,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    CHECK(model!=MAP_FAILED);
    bool selected_expert[E]={0};int32_t tmp[U];
    for(unsigned t=0;t<n;t++){route(tmp,rank,mode==6?2:(t+mode)%3);for(unsigned j=0;j<U;j++)selected_expert[tmp[j]]=true;}
    for(unsigned e=0;e<E;e++)if(selected_expert[e]) {
        for(unsigned row=0;row<H;row+=127) {
            unsigned b=(e+row)%160;
            mx *g=(mx*)(model+e*ge)+(size_t)row*160+b;
            mx *u=(mx*)(model+table+e*ge)+(size_t)row*160+b;
            g->e=u->e=116;memset(g->qs,0x22,16);memset(u->qs,0xaa,16);
        }
        for(unsigned row=0;row<M;row++) {
            unsigned b=(e+row)%72;
            mx *d=(mx*)(model+2*table+e*de)+(size_t)row*72+b;
            d->e=116;memset(d->qs,row&1?0xaa:0x22,16);
        }
    }
    size_t pairs=(size_t)n*U,nmid=pairs*H,ny=(size_t)n*M;
    int32_t *ids=malloc(pairs*4);float *weights=malloc(pairs*4),*x=malloc((size_t)n*K*4);
    float *initial=malloc(nmid*4),*got=malloc(ny*4),*observed=malloc(nmid*4);
    CHECK(ids&&weights&&x&&initial&&got&&observed);
    for(unsigned t=0;t<n;t++) {
        route(ids+(size_t)t*U,rank,mode==6?2:(t+mode)%3);
        for(unsigned j=0;j<U;j++)weights[(size_t)t*U+j]=(j+1)/16.f;
        for(unsigned k=0;k<K;k++)x[(size_t)t*K+k]=(t+k/32)&1?127.f:-127.f;
    }
    for(size_t i=0;i<nmid;i++)initial[i]=1234.f;
    ds4_gpu_set_deepseek41_model(true);CHECK(ds4_gpu_set_model_map(model,wb));
    int ssd_fd=-1;
    if(mode==6) {
        char path[]="/tmp/ds4-mxfp4-ssd-XXXXXX";
        ssd_fd=mkstemp(path);CHECK(ssd_fd>=0&&unlink(path)==0&&ftruncate(ssd_fd,(off_t)wb)==0);
        for(unsigned e=0;e<E;e++)if(selected_expert[e]) {
            CHECK(pwrite(ssd_fd,model+e*ge,ge,(off_t)(e*ge))==(ssize_t)ge);
            CHECK(pwrite(ssd_fd,model+table+e*ge,ge,(off_t)(table+e*ge))==(ssize_t)ge);
            CHECK(pwrite(ssd_fd,model+2*table+e*de,de,(off_t)(2*table+e*de))==(ssize_t)de);
        }
        CHECK(ds4_gpu_set_model_fd_for_map(ssd_fd,model));
        ds4_gpu_set_ssd_streaming(true);
        ds4_gpu_set_streaming_expert_cache_expert_bytes(2*ge+de);
        ds4_gpu_set_streaming_expert_cache_budget(12);
        CHECK(!mprotect(model,wb,PROT_NONE));
    } else {
        unsigned other=1-rank;
        CHECK(!mprotect(model+other*192*ge,192*ge,PROT_NONE));
        CHECK(!mprotect(model+table+other*192*ge,192*ge,PROT_NONE));
        CHECK(!mprotect(model+2*table+other*192*de,192*de,PROT_NONE));
    }
    ds4_gpu_tensor *it=upload(ids,pairs*4),*wt=upload(weights,pairs*4),*xt=upload(x,(size_t)n*K*4);
    ds4_gpu_tensor *gt=upload(initial,nmid*4),*ut=upload(initial,nmid*4),*mt=upload(initial,nmid*4);
    ds4_gpu_tensor *dt=upload(NULL,pairs*M*4),*ot=upload(NULL,ny*4);
    CHECK(it&&wt&&xt&&gt&&ut&&mt&&dt&&ot);
    if(mode==3||mode==4) {
        ids[0]=mode==3?-1:E;CHECK(ds4_gpu_tensor_write(it,0,ids,pairs*4));
        CHECK(!ds4_gpu_dsv41_routed_moe_typed_tp_tensor(ot,gt,ut,mt,dt,model,wb,0,table,2*table,
            it,wt,xt,n,rank,39,39,false,false));CHECK(sync_guards());puts("Invalid MXFP4 expert ID rejected PASS");goto done;
    }
    if(mode==6) {
        const ds4_gpu_stream_expert_table descriptor={model,wb,0,E,0,table,2*table,ge,de};
        if(n==1) {
            CHECK(ds4_gpu_routed_moe_set_selected_override(ids,U));
            CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&descriptor,ids,U));
            RUN(ds4_gpu_routed_moe_one_tensor(ot,gt,ut,mt,dt,model,wb,0,table,2*table,
                39,39,ge,K/32*17,de,H/32*17,K,H,M,it,wt,E,U,10.f,xt,NULL,0,false));
        } else {
            ds4_gpu_stream_expert_table odd=descriptor;odd.layer=1;
            CHECK(ds4_gpu_stream_expert_cache_reserve_layers(&descriptor,&odd));
            CHECK(ds4_gpu_stream_expert_cache_load_layer(&descriptor));
            bool half=false;
            RUN(ds4_gpu_routed_moe_batch_tensor(ot,gt,ut,mt,dt,model,wb,0,table,2*table,
                39,39,ge,K/32*17,de,H/32*17,K,H,M,it,wt,E,U,10.f,xt,0,n,&half,false));
            CHECK(!half&&ds4_gpu_stream_expert_cache_note_layer_consumed(&descriptor));
        }
        puts("MXFP4 SSD actual file reads; entire host weight mapping inaccessible");
    } else RUN(ds4_gpu_dsv41_routed_moe_typed_tp_tensor(ot,gt,ut,mt,dt,model,wb,0,table,2*table,
        it,wt,xt,n,rank,39,39,false,false));
    size_t failures=0;double worst=0;
    for(unsigned stage=(mode==6?2:0);stage<3;stage++) {
        CHECK(ds4_gpu_tensor_read(stage==0?gt:stage==1?ut:mt,0,observed,nmid*4));
        for(unsigned t=0;t<n;t++)for(unsigned j=0;j<U;j++)for(unsigned row=0;row<H;row++) {
            unsigned e=ids[(size_t)t*U+j];double expected=0;
            if(mode==6||e/192==rank) {
                double g=gate_ref(e,row,t,0),u=gate_ref(e,row,t,1);
                expected=stage==0?g:stage==1?u:g/(1+exp(-g))*u*weights[(size_t)t*U+j];
            }
            double error=fabs(observed[((size_t)t*U+j)*H+row]-expected);
            double tolerance=expected==0?0:3e-6*fmax(1.,fabs(expected));
            if(!isfinite(observed[((size_t)t*U+j)*H+row])||error>tolerance) {
                if(!failures)fprintf(stderr,"MXFP4 stage=%u t=%u e=%u row=%u got=%.9g expected=%.17g\n",stage,t,e,row,observed[((size_t)t*U+j)*H+row],expected);
                failures++;
            }
        }
    }
    /* Mid checked independently above. Quantize each observed Q8_K block in
     * double on CPU; sum only the 32 lanes of the single nonzero down block. */
    CHECK(ds4_gpu_tensor_read(ot,0,got,ny*4));
    if(mode==5)got[0]+=1.f; /* Negative oracle control must fail. */
    for(unsigned t=0;t<n;t++)for(unsigned row=0;row<M;row++) {
        double expected=0,l1=0;
        for(unsigned j=0;j<U;j++) {
            unsigned e=ids[(size_t)t*U+j];if(mode!=6&&e/192!=rank)continue;
            unsigned block=(e+row)%72,group=block/8;
            double maximum=0;
            for(unsigned k=group*256;k<(group+1)*256;k++)maximum=fmax(maximum,fabs(observed[((size_t)t*U+j)*H+k]));
            double scale=maximum/127.;
            for(unsigned k=block*32;k<(block+1)*32;k++) {
                double value=observed[((size_t)t*U+j)*H+k];
                double q=scale?nearbyint(value/scale):0;
                double term=q*scale*(row&1?-1.:1.)/2048.;expected+=term;l1+=fabs(term);
            }
        }
        double error=fabs(got[(size_t)t*M+row]-expected),tolerance=1e-7+3e-5*l1;
        worst=fmax(worst,error/tolerance);
        if(!isfinite(got[(size_t)t*M+row])||error>tolerance) {
            if(!failures)fprintf(stderr,"MXFP4 down t=%u row=%u got=%.9g expected=%.17g tol=%.9g\n",t,row,got[(size_t)t*M+row],expected,tolerance);
            failures++;
        }
    }
    fprintf(stderr,"MXFP4 TP rows=%u rank=%u mode=%u full_outputs=%zu failures=%zu worst_bound_fraction=%.9g\n",n,rank,mode,ny,failures,worst);
    CHECK(!failures&&sync_guards());
done:
    ds4_gpu_tensor_free(ot);ds4_gpu_tensor_free(dt);ds4_gpu_tensor_free(mt);ds4_gpu_tensor_free(ut);ds4_gpu_tensor_free(gt);
    ds4_gpu_tensor_free(xt);ds4_gpu_tensor_free(wt);ds4_gpu_tensor_free(it);ds4_gpu_cleanup();munmap(model,wb);
    if(ssd_fd>=0)close(ssd_fd);
    free(ids);free(weights);free(x);free(initial);free(got);free(observed);return 1;
}
int main(int argc,char **argv) {
    if(argc!=4)return 2;
    unsigned n=strtoul(argv[1],NULL,10),rank=strtoul(argv[2],NULL,10),mode=strtoul(argv[3],NULL,10);
    if(!n||n>2048||rank>1||mode>6)return 2;
    fprintf(stderr,"PID=%ld rows=%u rank=%u route-mode=%u\n",(long)getpid(),n,rank,mode);
    if(!ds4_gpu_init()||!run(n,rank,mode))return 1;
    puts("MXFP4 TP MoE case PASS");return 0;
}
