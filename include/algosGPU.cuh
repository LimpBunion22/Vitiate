#ifndef ALGOSGPU_CUH
#define ALGOSGPU_CUH

#include <cudaUtils.cuh>
#include <memoryGPU.cuh>

namespace gpu
{
    // adam
    class adam
    {
    private:
        int _size;
        float _momentum_beta;
        float _rms_beta;
        float *_momentum_data;
        float *_rms_data;

    private:
        void _free_mem();

    public:
        adam() = delete;
        adam(int size);
        adam(const adam &rh);
        adam(adam &&rh);
        adam &operator=(const adam &rh);
        adam &operator=(adam &&rh);
        ~adam();

        void set_momentum_beta(float momentum);
        void set_rms_beta(float rms);
        void algo(float *fx_accum, int size, int iteration, cudaStream_t stream); // recycle fx_accum as adam accumulator
    };

    // dropout
    class dropout
    {
    private:
        typedef struct
        {
            vtor mask;
            float keep_prob;
        } _dropout_layout;

    public:
        class receiver
        {
        public:
            virtual void on_dropout_receive(vtor mask) = 0;
            virtual void on_dropout_deactivate() = 0;
        };

    private:
        int _size;                                              // masks total size (same size as output layers)
        float *_data;                                           // masks data
        std::unordered_map<receiver *, _dropout_layout> _masks; // masks
        std::vector<receiver *> _masks_order;                   // keep masks ordered

    private:
        void _mask(vtor mask, float keep_prob, cudaStream_t stream);
        bool _allocate_mem();
        void _free_mem();

    public:
        dropout();
        dropout(const dropout &rh) = delete;
        dropout(dropout &&rh);
        dropout &operator=(const dropout &rh) = delete;
        dropout &operator=(dropout &&rh);
        ~dropout();

        void request_mask(receiver *rec, int v_size);
        void assign_masks();
        void free_masks();
        void update_masks(cudaStream_t stream);
    };

    // activations cero check
    class activations
    {
    private:
        cub_data *_cub = nullptr;
        float *_soft_accum = nullptr;
        int _sel_activation = net::RELU2; // default activation
        int _soft_size = 0;

    private:
        void _free_mem();
        void (activations::*_calc)(float *, int size, cudaStream_t) = &activations::_relu2;
        void (activations::*_calc_and_derive)(float *, float *, int size, cudaStream_t) = &activations::_fxrelu2;
        void (activations::*_soft)(float *, const float *, int size, cudaStream_t) = &activations::_copy;

        // calc
        void _sigmoid(float *inout, int size, cudaStream_t stream);
        void _relu(float *inout, int size, cudaStream_t stream);
        void _relu2(float *inout, int size, cudaStream_t stream);
        void _relu2_soft_max(float *inout, int size, cudaStream_t stream);

        // derive
        void _fxsigmoid(float *calc_inout, float *derive_out, int size, cudaStream_t stream);
        void _fxrelu(float *calc_inout, float *derive_out, int size, cudaStream_t stream);
        void _fxrelu2(float *calc_inout, float *derive_out, int size, cudaStream_t stream);
        void _fxrelu2_soft_max(float *calc_inout, float *derive_out, int size, cudaStream_t stream);

        // softmax
        void _soft_max(float *inout, const float *R, int size, cudaStream_t stream);
        void _copy(float *inout, const float *R, int size, cudaStream_t stream);

    public:
        activations() = default;
        activations(const activations &rh);
        activations(activations &&rh);
        activations &operator=(const activations &rh);
        activations &operator=(activations &&rh);
        ~activations();

        void select_activation(int activation, int soft_size);
        int get_activation() const;
        void set_cub(cub_data &cub);
        void calc(vtor inout, cudaStream_t stream);
        void calc_and_derive(vtor calc_inout, vtor derive_out, cudaStream_t stream);
        void soft_max(vtor inout, const vtor R, cudaStream_t stream);
    };

    // forward
    void forward(activations &activations, const vtor bias, const matrix params, vtor layer_output, const vtor prev_layer_output,
                 cublas_data &cublas, cudaStream_t stream);

    // async fwd can be safely executed in parallel as long as we use diff layer outputs buff, input buffs, output buffs and cub buffs per stream,
    // __restrict__ concurrent read is allowed and same cublas handle can be used for diff streams, with diff workspace per stream (but in same CPU thread).

    // specific gradient forward
    void gradient_forward(activations &activations, const vtor bias, const matrix params, vtor layer_output,
                          const vtor prev_layer_output, vtor fx_activations, const vtor mask, cublas_data &cublas, cudaStream_t stream);

    // gradient
    // reuse fx_params i+1 as fx_special_product_params (prev version), but accum must be updated for each layer
    // reuse layer_output(inner_vals) as tmp_gradient (prev version)
    void gradient_output(activations &activations, vtor R, const vtor output, vtor layer_output, const vtor prev_layer_output,
                         matrix fx_params, vtor fx_bias, const vtor fx_activations, cudaStream_t stream);
    // input case, prev_layer_output=input
    void gradient(vtor layer_output, const vtor next_layer_output, const vtor prev_layer_output, const matrix next_params, matrix fx_params,
                  matrix next_fx_params, vtor fx_bias, const vtor fx_activations, const vtor next_fx_activations, cublas_data &cublas, cudaStream_t stream);
    void update_partial_accum(float *fx_accum, float *fx_data, const matrix fx_params,
                              const vtor fx_bias, float *batch_multiplier, int batch_size, cublas_data &cublas, cudaStream_t stream);
    void reset_gradient_accum(float *fx_accum, int size, cudaStream_t stream);

}
#endif