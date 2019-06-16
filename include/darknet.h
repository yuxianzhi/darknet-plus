#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#ifdef GPU
    #define BLOCK 512

    #include <hip/hip_runtime.h>
    #include "rocrand/rocrand.h"
    #include "rocblas.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234
extern int gpu_index;

typedef struct{
    int classes = 0;
    char **names = NULL;
} metadata;

metadata get_metadata(char *file);

typedef struct{
    int *leaf = NULL;
    int n = 0;
    int *parent = NULL;
    int *child = NULL;
    int *group = NULL;
    char **name = NULL;

    int groups = 0;
    int *group_size = NULL;
    int *group_offset = NULL;
} tree;
tree *read_tree(char *filename);

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;

typedef struct{
    int batch = 0;
    float learning_rate = 0;
    float momentum = 0;
    float decay = 0;
    int adam = 0;
    float B1 = 0;
    float B2 = 0;
    float eps = 0;
    int t = 0;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    void (*forward_gpu)   (struct layer, struct network);
    void (*backward_gpu)  (struct layer, struct network);
    void (*update_gpu)    (struct layer, update_args);
    int batch_normalize = 0;
    int shortcut = 0;
    int batch = 0;
    int forced = 0;
    int flipped = 0;
    int inputs = 0;
    int outputs = 0;
    int nweights = 0;
    int nbiases = 0;
    int extra = 0;
    int truths = 0;
    int h=0,w=0,c=0;
    int out_h=0, out_w=0, out_c=0;
    int n = 0;
    int max_boxes = 0;
    int groups = 0;
    int size = 0;
    int side = 0;
    int stride = 0;
    int reverse = 0;
    int flatten = 0;
    int spatial = 0;
    int pad = 0;
    int sqrt = 0;
    int flip = 0;
    int index = 0;
    int binary = 0;
    int xnor = 0;
    int steps = 0;
    int hidden = 0;
    int truth = 0;
    float smooth = 0;
    float dot = 0;
    float angle = 0;
    float jitter = 0;
    float saturation = 0;
    float exposure = 0;
    float shift = 0;
    float ratio = 0;
    float learning_rate_scale = 0;
    float clip = 0;
    int noloss = 0;
    int softmax = 0;
    int classes = 0;
    int coords = 0;
    int background = 0;
    int rescore = 0;
    int objectness = 0;
    int joint = 0;
    int noadjust = 0;
    int reorg = 0;
    int log = 0;
    int tanh = 0;
    int *mask = NULL;
    int total = 0;

    float alpha = 0;
    float beta = 0;
    float kappa = 0;

    float coord_scale = 0;
    float object_scale = 0;
    float noobject_scale = 0;
    float mask_scale = 0;
    float class_scale = 0;
    int bias_match = 0;
    int random = 0;
    float ignore_thresh = 0;
    float truth_thresh = 0;
    float thresh = 0;
    float focus = 0;
    int classfix = 0;
    int absolute = 0;

    int onlyforward = 0;
    int stopbackward = 0;
    int dontload = 0;
    int dontsave = 0;
    int dontloadscales = 0;
    int numload = 0;

    float temperature = 0;
    float probability = 0;
    float scale = 0;

    char  * cweights = NULL;
    int   * indexes = NULL;
    int   * input_layers = NULL;
    int   * input_sizes = NULL;
    int   * map = NULL;
    int   * counts = NULL;
    float ** sums = NULL;
    float * rand = NULL;
    float * cost = NULL;
    float * state = NULL;
    float * prev_state = NULL;
    float * forgot_state = NULL;
    float * forgot_delta = NULL;
    float * state_delta = NULL;
    float * combine_cpu = NULL;
    float * combine_delta_cpu = NULL;

    float * concat = NULL;
    float * concat_delta = NULL;

    float * binary_weights = NULL;

    float * biases = NULL;
    float * bias_updates = NULL;

    float * scales = NULL;
    float * scale_updates = NULL;

    float * weights = NULL;
    float * weight_updates = NULL;

    float * delta = NULL;
    float * output = NULL;
    float * loss = NULL;
    float * squared = NULL;
    float * norms = NULL;

    float * spatial_mean = NULL;
    float * mean = NULL;
    float * variance = NULL;

    float * mean_delta = NULL;
    float * variance_delta = NULL;

    float * rolling_mean = NULL;
    float * rolling_variance = NULL;

    float * x = NULL;
    float * x_norm = NULL;

    float * m = NULL;
    float * v = NULL;
    
    float * bias_m = NULL;
    float * bias_v = NULL;
    float * scale_m = NULL;
    float * scale_v = NULL;


    float *z_cpu = NULL;
    float *r_cpu = NULL;
    float *h_cpu = NULL;
    float * prev_state_cpu = NULL;

    float *temp_cpu = NULL;
    float *temp2_cpu = NULL;
    float *temp3_cpu = NULL;

    float *dh_cpu = NULL;
    float *hh_cpu = NULL;
    float *prev_cell_cpu = NULL;
    float *cell_cpu = NULL;
    float *f_cpu = NULL;
    float *i_cpu = NULL;
    float *g_cpu = NULL;
    float *o_cpu = NULL;
    float *c_cpu = NULL;
    float *dc_cpu = NULL; 

    float * binary_input = NULL;

    struct layer *input_layer = NULL;
    struct layer *self_layer = NULL;
    struct layer *output_layer = NULL;

    struct layer *reset_layer = NULL;
    struct layer *update_layer = NULL;
    struct layer *state_layer = NULL;

    struct layer *input_gate_layer = NULL;
    struct layer *state_gate_layer = NULL;
    struct layer *input_save_layer = NULL;
    struct layer *state_save_layer = NULL;
    struct layer *input_state_layer = NULL;
    struct layer *state_state_layer = NULL;

    struct layer *input_z_layer = NULL;
    struct layer *state_z_layer = NULL;

    struct layer *input_r_layer = NULL;
    struct layer *state_r_layer = NULL;

    struct layer *input_h_layer = NULL;
    struct layer *state_h_layer = NULL;
	
    struct layer *wz = NULL;
    struct layer *uz = NULL;
    struct layer *wr = NULL;
    struct layer *ur = NULL;
    struct layer *wh = NULL;
    struct layer *uh = NULL;
    struct layer *uo = NULL;
    struct layer *wo = NULL;
    struct layer *uf = NULL;
    struct layer *wf = NULL;
    struct layer *ui = NULL;
    struct layer *wi = NULL;
    struct layer *ug = NULL;
    struct layer *wg = NULL;

    tree *softmax_tree = NULL;

    size_t workspace_size = 0;

#ifdef GPU
    int *indexes_gpu = NULL;

    float *z_gpu = NULL;
    float *r_gpu = NULL;
    float *h_gpu = NULL;

    float *temp_gpu = NULL;
    float *temp2_gpu = NULL;
    float *temp3_gpu = NULL;

    float *dh_gpu = NULL;
    float *hh_gpu = NULL;
    float *prev_cell_gpu = NULL;
    float *cell_gpu = NULL;
    float *f_gpu = NULL;
    float *i_gpu = NULL;
    float *g_gpu = NULL;
    float *o_gpu = NULL;
    float *c_gpu = NULL;
    float *dc_gpu = NULL; 

    float *m_gpu = NULL;
    float *v_gpu = NULL;
    float *bias_m_gpu = NULL;
    float *scale_m_gpu = NULL;
    float *bias_v_gpu = NULL;
    float *scale_v_gpu = NULL;

    float * combine_gpu = NULL;
    float * combine_delta_gpu = NULL;

    float * prev_state_gpu = NULL;
    float * forgot_state_gpu = NULL;
    float * forgot_delta_gpu = NULL;
    float * state_gpu = NULL;
    float * state_delta_gpu = NULL;
    float * gate_gpu = NULL;
    float * gate_delta_gpu = NULL;
    float * save_gpu = NULL;
    float * save_delta_gpu = NULL;
    float * concat_gpu = NULL;
    float * concat_delta_gpu = NULL;

    float * binary_input_gpu = NULL;
    float * binary_weights_gpu = NULL;

    float * mean_gpu = NULL;
    float * variance_gpu = NULL;

    float * rolling_mean_gpu = NULL;
    float * rolling_variance_gpu = NULL;

    float * variance_delta_gpu = NULL;
    float * mean_delta_gpu = NULL;

    float * x_gpu = NULL;
    float * x_norm_gpu = NULL;
    float * weights_gpu = NULL;
    float * weight_updates_gpu = NULL;
    float * weight_change_gpu = NULL;

    float * biases_gpu = NULL;
    float * bias_updates_gpu = NULL;
    float * bias_change_gpu = NULL;

    float * scales_gpu = NULL;
    float * scale_updates_gpu = NULL;
    float * scale_change_gpu = NULL;

    float * output_gpu = NULL;
    float * loss_gpu = NULL;
    float * delta_gpu = NULL;
    float * rand_gpu = NULL;
    float * squared_gpu = NULL;
    float * norms_gpu = NULL;
#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

void free_layer(layer);

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    int n = 0;
    int batch = 0;
    size_t *seen = NULL;
    int *t = NULL;
    float epoch = 0;
    int subdivisions = 0;
    layer *layers = NULL;
    float *output = NULL;
    learning_rate_policy policy;

    float learning_rate = 0;
    float momentum = 0;
    float decay = 0;
    float gamma = 0;
    float scale = 0;
    float power = 0;
    int time_steps = 0;
    int step = 0;
    int max_batches = 0;
    float *scales = NULL;
    int   *steps = NULL;
    int num_steps = 0;
    int burn_in = 0;

    int adam = 0;
    float B1 = 0;
    float B2 = 0;
    float eps = 0;

    int inputs = 0;
    int outputs = 0;
    int truths = 0;
    int notruth = 0;
    int h=0, w=0, c=0;
    int max_crop = 0;
    int min_crop = 0;
    float max_ratio = 0;
    float min_ratio = 0;
    int center = 0;
    float angle = 0;
    float aspect = 0;
    float exposure = 0;
    float saturation = 0;
    float hue = 0;
    int random = 0;

    int gpu_index =0;
    tree *hierarchy = NULL;

    float *input = NULL;
    float *truth = NULL;
    float *delta = NULL;
    float *workspace = NULL;
    int train = 0;
    int index = 0;
    float *cost = NULL;
    float clip = 0;

#ifdef GPU
    float *input_gpu = NULL;
    float *truth_gpu = NULL;
    float *delta_gpu = NULL;
    float *output_gpu = NULL;
#endif

} network;

typedef struct {
    int w = 0;
    int h = 0;
    float scale = 0;
    float rad = 0;
    float dx = 0;
    float dy = 0;
    float aspect = 0;
} augment_args;

typedef struct {
    int w = 0;
    int h = 0;
    int c = 0;
    float *data = NULL;
} image;

typedef struct{
    float x = 0;
    float y = 0;
    float w = 0;
    float h = 0;
} box;

typedef struct detection{
    box bbox;
    int classes = 0;
    float *prob = NULL;
    float *mask = NULL;
    float objectness = 0;
    int sort_class = 0;
} detection;

typedef struct matrix{
    int rows = 0, cols = 0;
    float **vals = NULL;
} matrix;


typedef struct{
    int w=0, h=0;
    matrix X;
    matrix y;
    int shallow = 0;
    int *num_boxes = NULL;
    box **boxes = NULL;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
} data_type;

typedef struct load_args{
    int threads = 0;
    char **paths = NULL;
    char *path = NULL;
    int n = 0;
    int m = 0;
    char **labels = NULL;
    int h = 0;
    int w = 0;
    int out_w = 0;
    int out_h = 0;
    int nh = 0;
    int nw = 0;
    int num_boxes = 0;
    int min=0, max=0, size=0;
    int classes = 0;
    int background = 0;
    int scale = 0;
    int center = 0;
    int coords = 0;
    float jitter = 0;
    float angle = 0;
    float aspect = 0;
    float saturation = 0;
    float exposure = 0;
    float hue = 0;
    data *d = NULL;
    image *im = NULL;
    image *resized = NULL;
    data_type type;
    tree *hierarchy = NULL;
} load_args;

typedef struct{
    int id = 0;
    float x=0,y=0,w=0,h=0;
    float left=0, right=0, top=0, bottom=0;
} box_label;


network *load_network(char *cfg, char *weights, int clear);
load_args get_base_args(network *net);

void free_data(data d);

typedef struct node{
    void *val = NULL;
    struct node *next = NULL;
    struct node *prev = NULL;
} node;

typedef struct list{
    int size = 0;
    node *front = NULL;
    node *back = NULL;
} list;

pthread_t load_data(load_args args);
list *read_data_cfg(char *filename);
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);
data resize_data(data orig, int w, int h);
data *tile_data(data orig, int divs, int size);
data select_data(data *orig, int *inds);

void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);


float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax(float *input, int n, float temp, int stride, float *output);

int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void fill_gpu(int N, float ALPHA, float * X, int INCX);
void scal_gpu(int N, float ALPHA, float * X, int INCX);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);

void hip_set_device(int n);
void hip_free(float *x_gpu);
float *hip_make_array(float *x, size_t n);
void hip_pull_array(float *x_gpu, float *x, size_t n);
float hip_mag_array(float *x_gpu, size_t n);
void hip_push_array(float *x_gpu, float *x, size_t n);

void forward_network_gpu(network *net);
void backward_network_gpu(network *net);
void update_network_gpu(network *net);

float train_networks(network **nets, int n, data d, int interval);
void sync_nets(network **nets, int n, int interval);
void harmless_update_network_gpu(network *net);
#endif
image get_label(image **characters, char *string, int size);
void draw_label(image a, int r, int c, image label, const float *rgb);
void save_image(image im, const char *name);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void grayscale_image_3c(image im);
void normalize_image(image p);
void matrix_to_csv(matrix m);
float train_network_sgd(network *net, data d, int n);
void rgbgr_image(image im);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_cifar10_data(char *filename);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix csv_to_matrix(char *filename);
float *network_accuracies(network *net, data d, int n);
float train_network_datum(network *net);
image make_random_image(int w, int h, int c);

void denormalize_connected_layer(layer l);
void denormalize_convolutional_layer(layer l);
void statistics_connected_layer(layer l);
void rescale_weights(layer l, float scale, float trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen);
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);

network *parse_network_cfg(char *filename);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);

void zero_objectness(layer l);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
void free_network(network *net);
void set_batch_network(network *net, int b);
void set_temp_network(network *net, float t);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void censor_image(image im, int dx, int dy, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, float thresh);
image mask_to_rgb(image mask);
int resize_network(network *net, int w, int h);
void free_matrix(matrix m);
void test_resize(char *filename);
int show_image(image p, const char *name, int ms);
image copy_image(image p);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
float get_current_rate(network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
size_t get_current_batch(network *net);
void constrain_image(image im);
image get_network_image_layer(network *net, int i);
layer get_network_output_layer(network *net);
void top_predictions(network *net, int n, int *index);
void flip_image(image a);
image float_to_image(int w, int h, int c, float *data);
void ghost_image(image source, image dest, int dx, int dy);
float network_accuracy(network *net, data d);
void random_distort_image(image im, float hue, float saturation, float exposure);
void fill_image(image m, float s);
image grayscale_image(image im);
void rotate_image_cw(image im, int times);
double what_time_is_it_now();
image rotate_image(image m, float rad);
void visualize_network(network *net);
float box_iou(box a, box b);
data load_all_cifar10();
box_label *read_boxes(char *filename, int *n);
box float_to_box(float *f, int stride);
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

matrix network_predict_data(network *net, data test);
image **load_alphabet();
image get_network_image(network *net);
float *network_predict(network *net, float *input);

int network_width(network *net);
int network_height(network *net);
float *network_predict_image(network *net, image im);
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);

void reset_network_state(network *net, int b);

char **get_labels(char *filename);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh);

matrix make_matrix(int rows, int cols);

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
void make_window(char *name, int w, int h, int fullscreen);
#endif

void free_image(image m);
float train_network(network *net, data d);
pthread_t load_data_in_thread(load_args args);
void load_data_blocking(load_args args);
list *get_paths(char *filename);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
float sec(clock_t clocks);
void **list_to_array(list *l);
void top_k(float *a, int n, int k, int *index);
int *read_map(char *filename);
void error(const char *s);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
int *random_index_order(int min, int max);
void free_list(list *l);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
void scale_array(float *a, int n, float s);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
float rand_normal();
float rand_uniform(float min, float max);

#ifdef __cplusplus
}
#endif
#endif
