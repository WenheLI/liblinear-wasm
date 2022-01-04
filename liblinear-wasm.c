#include "emscripten.h"
#include "liblinear/linear.h"
#include <stdlib.h>
#include<stdio.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

typedef struct parameter parameter;
typedef struct problem problem;
typedef struct feature_node feature_node;
typedef struct model model;
typedef struct feature_node feature_node;


EMSCRIPTEN_KEEPALIVE
parameter* prepare_param(int solver_type, double C, double eps, int nr_weight, int* weight_label, 
                                double* weight, double p, double nu, double* init_sol, int regularize_bias) {
  parameter* param = Malloc(struct parameter, 1);
  param->solver_type = solver_type;
  param->C = C;
  param->eps = eps;
  param->nr_weight = nr_weight;
  param->weight_label = weight_label;
  param->weight = weight;
  param->p = p;
  param->nu = nu;
  param->init_sol = init_sol;
  param->regularize_bias = regularize_bias;
  return param;
}

EMSCRIPTEN_KEEPALIVE
void free_param(parameter* param) {
  if (param->weight_label) free(param->weight_label);
  if (param->weight) free(param->weight);
  if (param->init_sol) free(param->init_sol);
  free(param);
}

problem* init_problem_internal(int num_data, int num_feat) {
  problem* prob = Malloc(problem, 1);
  prob->l = num_data;
  prob->x = Malloc(feature_node*, num_data);
  prob->y = Malloc(double, num_data);
  feature_node* x_space = Malloc(feature_node, (num_feat + 2) * num_data);
  for (int i = 0; i < num_data; i++) {
      prob->x[i] = x_space + i * (num_feat + 2);
  }
  return prob;
}

EMSCRIPTEN_KEEPALIVE
problem* init_problem(double* data, double* labels, int num_data, int num_feat, int bias) {
  problem* prob = init_problem_internal(num_data, num_feat);
  prob->bias = bias;
  for (int i = 0; i < num_data; i++) {
    for (int j = 0; j < num_feat; j++) {
      prob->x[i][j].index = j + 1;
      prob->x[i][j].value = data[i * num_feat + j];
    }
    prob->y[i] = labels[i];
    if (prob->bias >= 0) {
      prob->x[i][num_feat].value = prob->bias;
      prob->x[i][num_feat].index = num_feat + 1;
      prob->x[i][num_feat + 1].index = -1;
    } else {
      prob->x[i][num_feat].index = -1;
    }
  }

  if (prob->bias >= 0) {
    prob->n = num_feat + 1;
    for (int i = 1; i < prob->l; i++) {
      (prob->x[i]-2)->index = prob->n;
    }
    prob->x[(num_feat + 1) * num_data - 2]->index = prob->n;
  } else {
    prob->n = num_feat;
  }

  return prob;
}

EMSCRIPTEN_KEEPALIVE
void free_problem(problem* prob) {
  free(prob->x);
  free(prob->y);
  free(prob);
}


EMSCRIPTEN_KEEPALIVE
model* train_model(problem* prob, parameter* param) {
  model* model = train(prob, param);
  return model;
}

EMSCRIPTEN_KEEPALIVE
void free_model(model* model) {
  free_and_destroy_model(&model);
}

feature_node* construct_node(double* data, model* model, int num_feat) {
  feature_node* x = Malloc(feature_node, (num_feat + 2));
  for (int i = 0; i < num_feat; i++) {
    x[i].index = i + 1;
    x[i].value = data[i];
  }
  if (model->bias >= 0) {
    x[num_feat].value = model->bias;
    x[num_feat].index = num_feat + 1;
    x[num_feat + 1].index = -1;
  } else {
    x[num_feat].index = -1;
  }
  return x;
}

EMSCRIPTEN_KEEPALIVE
double predict_one(model* model, double* data, int num_feat) {
  feature_node* x = construct_node(data, model, num_feat);
  double pred = predict(model, x);
  return pred;
}

EMSCRIPTEN_KEEPALIVE
double* predict_one_prob(model* model, double* data) {
  feature_node* x = construct_node(data, model, model->nr_class);
  double* prob = Malloc(double, model->nr_class);
  predict_probability(model, x, prob);
  return prob;
}
