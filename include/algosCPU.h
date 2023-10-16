#ifndef ALGOSCPU_H
#define ALGOSCPU_H

#include <memoryCPU.h>
#include <defines.h>

namespace cpu::single
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
        void algo(float *__restrict fx_accum, int size, int iteration); // recycle fx_accum as adam accumulator
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
        void _mask(vtor mask, float keep_prob);
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
        void update_masks();
    };

    // activations cero check
    class activations
    {
    private:
        float *__restrict _soft_accum = nullptr;
        int _sel_activation = net::RELU2; // default activation
        int _soft_size = 0;

    private:
        void _free_mem();
        void (activations::*_soft)(float *__restrict, const float *__restrict,
                                   int size) = &activations::_copy;
        void (activations::*_calc)(float *__restrict, const float *__restrict, const float *__restrict,
                                   const float *__restrict, int, int) = &activations::_sgemv_relu2;
        void (activations::*_calc_and_derive)(float *__restrict, float *__restrict, const float *__restrict,
                                              const float *__restrict, const float *__restrict, int, int) = &activations::_fx_sgemv_relu2;

        // calc
        float _sigmoid(float in); // all check for cero/bound limits if needed
        float _relu(float in);
        float _relu2(float in);
        float _relu2_soft_max(float in);
        void _sgemv_sigmoid(float *__restrict inout, const float *__restrict a, const float *__restrict b,
                            const float *__restrict c, int rows, int cols);
        void _sgemv_relu(float *__restrict inout, const float *__restrict a, const float *__restrict b,
                         const float *__restrict c, int rows, int cols);
        void _sgemv_relu2(float *__restrict inout, const float *__restrict a, const float *__restrict b,
                          const float *__restrict c, int rows, int cols);
        void _sgemv_relu2_soft_max(float *__restrict inout, const float *__restrict a, const float *__restrict b,
                                   const float *__restrict c, int rows, int cols);

        // calc and derive
        float _fxsigmoid(float sigmoid);
        float _fxrelu(float in);
        float _fxrelu2(float in);
        void _fx_sgemv_sigmoid(float *__restrict calc_out, float *__restrict derive_out, const float *__restrict a,
                               const float *__restrict b, const float *__restrict c, int rows, int cols);
        void _fx_sgemv_relu(float *__restrict calc_out, float *__restrict derive_out, const float *__restrict a,
                            const float *__restrict b, const float *__restrict c, int rows, int cols);
        void _fx_sgemv_relu2(float *__restrict calc_out, float *__restrict derive_out, const float *__restrict a,
                             const float *__restrict b, const float *__restrict c, int rows, int cols);
        void _fx_sgemv_relu2_soft_max(float *__restrict calc_out, float *__restrict derive_out, const float *__restrict a,
                                      const float *__restrict b, const float *__restrict c, int rows, int cols);

        // softmax
        void _soft_max(float *__restrict inout, const float *__restrict R, int size);
        void _copy(float *__restrict dst, const float *__restrict R, int size);

    public:
        activations() = default;
        activations(const activations &rh);
        activations(activations &&rh);
        activations &operator=(const activations &rh);
        activations &operator=(activations &&rh);
        ~activations();

        void select_activation(int activation, int soft_size);
        int get_activation() const;
        void calc(vtor inout, const matrix a, const vtor b, const vtor c);
        void calc_and_derive(vtor calc_out, vtor derive_out, const matrix a, const vtor b, const vtor c);
        void soft_max(vtor inout, const vtor R);
    };

    // gradient
    // reuse fx_params i+1 as fx_special_product_params (prev version), but accum must be updated for each layer
    // reuse layer_output(inner_vals) as tmp_gradient (prev version)
    void gradient_output(activations &activations, vtor R, const vtor output, const vtor layer_output,
                         const vtor prev_layer_output, matrix fx_params, vtor fx_bias, const vtor fx_activations);
    // input case, prev_layer_output=input
    void gradient(vtor layer_output, const vtor next_layer_output, const vtor prev_layer_output, const matrix next_params,
                  matrix fx_params, matrix next_fx_params, vtor fx_bias, const vtor fx_activations, const vtor next_fx_activations);
    void update_partial_accum(float *__restrict fx_accum, float *__restrict fx_data, const matrix fx_params, const vtor fx_bias, int batch_size);
    void reset_gradient_accum(float *__restrict fx_accum, int size);
}

#endif