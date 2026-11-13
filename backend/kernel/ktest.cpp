// ===========================================================================
// ktest.cpp — kernel 对拍驱动：golden safetensors → GPU kernel → 统计判分
// ===========================================================================
//
// 跑法（backend/ 目录下，同 main.cpp 的相对路径约定）：
//   make ktest            # = build/ktest ../verify/full_out/prefill.safetensors
//   build/ktest <file>    # 换 decode/cache 快照亦同构
//
// 对拍判分制度（与 tools/cmp_bin.py 头注同源，别用错档）：
//   golden 计算全程 f32、落盘压 bf16，故 kernel f32 输出对 golden 只能到
//   bf16 舍入量级——判 max|diff| ≤ atol_max（½ ulp_bf16(|y|max) × 余量）+
//   mean|diff| ≤ atol_mean；另报 f32→bf16 后与 golden 逐位相等率（观察项，
//   舍入边界翻转属正常）。合成用例（无落盘舍入）走 f32 紧公差，抓索引/
//   分组类的低级错。
//
// safetensors 读取是最小实现：8B 头长 + JSON 头 + 数据区，BF16 高 16 位
// 直落 f32（同 tools/st.py），JSON 解析用 opt/nlohmann。
// ===========================================================================
#include "cuda_kernels.cuh"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// ---- 最小 safetensors 读取 ------------------------------------------------------

namespace {

struct StTensor {
    std::vector<int64_t> shape;
    std::string dtype;         // "BF16" / "F32" / ...
    const uint8_t* data;       // 指向文件缓冲内（StFile 活着则有效）
    size_t bytes;
};

struct StFile {
    explicit StFile(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "ktest: can't open %s\n", path.c_str());
            std::exit(1);
        }
        f.seekg(0, std::ios::end);
        buf.resize((size_t)f.tellg());
        f.seekg(0);
        f.read((char*)buf.data(), buf.size());
        const uint64_t hlen = *reinterpret_cast<const uint64_t*>(buf.data());
        hdr = nlohmann::json::parse(buf.data() + 8, buf.data() + 8 + hlen);
        base = 8 + hlen;
    }

    // 找不到 key 直接报错退出（对拍 key 写错属开发期 bug，fail fast）
    StTensor get(const std::string& key) const {
        auto it = hdr.find(key);
        if (it == hdr.end()) {
            std::fprintf(stderr, "ktest: key not found: %s\n", key.c_str());
            std::exit(1);
        }
        StTensor t;
        t.dtype = (*it)["dtype"];
        for (auto& d : (*it)["shape"]) t.shape.push_back(d.get<int64_t>());
        const size_t s = (*it)["data_offsets"][0], e = (*it)["data_offsets"][1];
        t.data = buf.data() + base + s;
        t.bytes = e - s;
        return t;
    }

    std::vector<uint8_t> buf;
    nlohmann::json hdr;
    size_t base = 0;
};

// 浮点张量统一转 f32（BF16 高 16 位直落；非浮点报错——对拍不涉整型激活）
std::vector<float> as_f32(const StTensor& t) {
    const size_t n = t.bytes / 2;
    std::vector<float> v;
    if (t.dtype == "BF16") {
        v.resize(n);
        const uint16_t* u = reinterpret_cast<const uint16_t*>(t.data);
        for (size_t i = 0; i < n; i++) {  // 高 16 位直落 f32（同 tools/st.py）
            const uint32_t b = (uint32_t)u[i] << 16;
            std::memcpy(&v[i], &b, 4);
        }
    } else if (t.dtype == "F32") {
        v.resize(t.bytes / 4);
        std::memcpy(v.data(), t.data, t.bytes);
    } else {
        std::fprintf(stderr, "ktest: unsupported dtype %s\n", t.dtype.c_str());
        std::exit(1);
    }
    return v;
}

size_t numel(const StTensor& t) {
    size_t n = 1;
    for (auto d : t.shape) n *= (size_t)d;
    return n;
}

// ---- GPU 小助手 -------------------------------------------------------------------

void* dev_alloc(const void* h, size_t bytes) {
    void* d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) {
        std::fprintf(stderr, "ktest: cudaMalloc failed\n");
        std::exit(1);
    }
    if (h && cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "ktest: H2D failed\n");
        std::exit(1);
    }
    return d;
}

// ---- 判分 ------------------------------------------------------------------------

struct DiffStats {
    double max_abs, mean_abs;
    long bf16_eq;      // kernel f32→bf16 后与 golden 逐位相等数
    size_t n;
};

uint16_t f32_to_bf16_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    // 就近舍入到 bf16（round-to-nearest-even，同 __float2bfloat16）
    uint32_t r = (u >> 16) & 1;
    r += 0x7fff;
    return (uint16_t)((u + r) >> 16);
}

// golden 对照判分：atol 取 bf16 舍入量级（ktest 头注的判分制度）
DiffStats compare(const std::vector<float>& got, const std::vector<float>& want,
                  const char* tag, double atol_max, double atol_mean) {
    DiffStats s{0, 0, 0, got.size()};
    double sum = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const double d = std::fabs((double)got[i] - (double)want[i]);
        s.max_abs = std::max(s.max_abs, d);
        sum += d;
        if (f32_to_bf16_bits(got[i]) == f32_to_bf16_bits(want[i])) s.bf16_eq++;
    }
    s.mean_abs = sum / got.size();
    const bool pass = s.max_abs <= atol_max && s.mean_abs <= atol_mean;
    std::printf("  %-52s max=%.3e mean=%.3e bf16_eq=%zu/%zu(%.1f%%)  %s\n", tag,
                s.max_abs, s.mean_abs, (size_t)s.bf16_eq, s.n,
                100.0 * s.bf16_eq / s.n, pass ? "PASS" : "FAIL");
    return s;
}

int g_fail = 0;

// ---- RMSNorm 对拍 ----------------------------------------------------------------

// golden 用例：in/out key + 分组宽。w 恒 nullptr（golden 的 RMSNorm 权重
// 全 0 → 乘 1，见 rmsnorm.cu 头注）。
void rmsnorm_golden(const StFile& st, const std::string& in_key,
                    const std::string& out_key, int group) {
    const StTensor xin = st.get(in_key), yout = st.get(out_key);
    const size_t width = (size_t)xin.shape.back();
    const int rows = (int)(numel(xin) / width);
    const std::vector<float> x = as_f32(xin), want = as_f32(yout);
    if (numel(yout) != numel(xin)) {
        std::printf("  %-52s shape mismatch\n", in_key.c_str());
        g_fail++;
        return;
    }

    float *dx = (float*)dev_alloc(x.data(), x.size() * 4), *dy = (float*)dev_alloc(nullptr, x.size() * 4);
    kern::rmsnorm(dx, nullptr, dy, nullptr, rows, (int)width, group, 1e-6f, nullptr);
    std::vector<float> got(x.size());
    if (cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "ktest: D2H/kernel error\n");
        std::exit(1);
    }
    cudaFree(dx);
    cudaFree(dy);

    const DiffStats s = compare(got, want, in_key.c_str(), 0.03, 2e-3);
    if (!(s.max_abs <= 0.03 && s.mean_abs <= 2e-3)) g_fail++;
}

// 合成用例：手造 (rows=3, width=8, group=4)，w 非空（抓跨段 w 索引）、
// y16 通路（bf16 像位型核对）。期望值 host 朴素重算，紧公差 1e-6。
void rmsnorm_synthetic() {
    const int rows = 3, width = 8, group = 4;
    std::vector<float> x(rows * width), w(width), want(rows * width);
    for (int i = 0; i < rows * width; i++) x[i] = 0.1f * (float)(i + 1) - 0.9f;
    for (int i = 0; i < width; i++) w[i] = 0.25f * (float)i;
    for (int r = 0; r < rows; r++)
        for (int g = 0; g < width / group; g++) {
            double ss = 0;
            for (int i = 0; i < group; i++) ss += (double)x[r * width + g * group + i] * x[r * width + g * group + i];
            const float rms = (float)(1.0 / std::sqrt(ss / group + 1e-6));
            for (int i = 0; i < group; i++)
                want[r * width + g * group + i] =
                    x[r * width + g * group + i] * rms * (1.f + w[g * group + i]);
        }

    float* dx = (float*)dev_alloc(x.data(), x.size() * 4);
    float* dw = (float*)dev_alloc(w.data(), w.size() * 4);
    float* dy = (float*)dev_alloc(nullptr, want.size() * 4);
    uint16_t* dy16 = (uint16_t*)dev_alloc(nullptr, want.size() * 2);
    kern::rmsnorm(dx, dw, dy, dy16, rows, width, group, 1e-6f, nullptr);
    std::vector<float> got(want.size());
    std::vector<uint16_t> got16(want.size());
    cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got16.data(), dy16, got16.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dw); cudaFree(dy); cudaFree(dy16);

    double mx = 0;
    for (size_t i = 0; i < got.size(); i++) mx = std::max(mx, std::fabs((double)got[i] - want[i]));
    bool ok = mx <= 1e-6;
    for (size_t i = 0; i < got16.size() && ok; i++)          // bf16 像 = f32 值就近舍入
        ok = got16[i] == f32_to_bf16_bits(got[i]);
    std::printf("  %-52s max=%.3e y16_bits=%s  %s\n", "synthetic(3,8,g4,w≠0)", mx,
                ok ? "ok" : "bad", ok ? "PASS" : "FAIL");
    if (!ok) g_fail++;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "../verify/full_out/prefill.safetensors";
    std::printf("== kern::rmsnorm ==  golden: %s\n", path.c_str());
    const StFile st(path);

    // 分组宽 2560：attn/mlp 超连接 hc_norm + 顶层 mixer（layer0 特例：输入
    // 是 embed 广播成的 4 条全同流，也顺带覆盖"每行内 4 段"的打平）
    rmsnorm_synthetic();
    rmsnorm_golden(st, "prefill/model.layers.0/attn_hyper_connection.hc_norm/in[0]",
                   "prefill/model.layers.0/attn_hyper_connection.hc_norm/out", 2560);
    rmsnorm_golden(st, "prefill/model.layers.5/attn_hyper_connection.hc_norm/in[0]",
                   "prefill/model.layers.5/attn_hyper_connection.hc_norm/out", 2560);
    rmsnorm_golden(st, "prefill/model.layers.3/attn_hyper_connection.hc_norm/in[0]",   // QSA 层
                   "prefill/model.layers.3/attn_hyper_connection.hc_norm/out", 2560);
    rmsnorm_golden(st, "prefill/model.layers.5/mlp_hyper_connection.hc_norm/in[0]",
                   "prefill/model.layers.5/mlp_hyper_connection.hc_norm/out", 2560);
    rmsnorm_golden(st, "prefill/model.hyper_connection_mixer/hc_norm/in[0]",
                   "prefill/model.hyper_connection_mixer/hc_norm/out", 2560);
    // 逐头 norm：width=group（q 24 头 / k 2 头 / indexer 4 头 ×128、池化键 16×128）
    rmsnorm_golden(st, "prefill/model.layers.3/self_attn.q_norm/in[0]",
                   "prefill/model.layers.3/self_attn.q_norm/out", 256);
    rmsnorm_golden(st, "prefill/model.layers.3/self_attn.k_norm/in[0]",
                   "prefill/model.layers.3/self_attn.k_norm/out", 256);
    rmsnorm_golden(st, "prefill/model.layers.3/self_attn.indexer.q_layernorm/in[0]",
                   "prefill/model.layers.3/self_attn.indexer.q_layernorm/out", 128);
    rmsnorm_golden(st, "prefill/model.layers.3/self_attn.indexer.k_layernorm/in[0]",
                   "prefill/model.layers.3/self_attn.indexer.k_layernorm/out", 128);

    std::printf(g_fail ? "== %d FAIL ==\n" : "== all PASS ==\n", g_fail);
    return g_fail ? 1 : 0;
}
