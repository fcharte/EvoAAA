#!/usr/bin/env Rscript
require("ruta", quietly = TRUE)
# Chromosome codification

ae_type <-
  c("Basic AE",
    "Denoising AE",
    "Contractive AE",
    "Robust AE",
    "Sparse AE",
    "Variational AE")
ae_layer_activation <-
  c("linear",
    "sigmoid",
    "tanh",
    "relu",
    "selu",
    "elu",
    "softplus",
    "softsign")
ae_output_activation <-
  c("linear", "relu", "elu", "softplus") # positive unbounded functions
ae_loss <-
  c(
    "mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "binary_crossentropy",
    "cosine_proximity"
  )

gen_ae_type   <- 1
gen_ae_layers <- 2
gen_ae_coding_length <- 6
gen_ae_coding_activation <- 10
gen_ae_output_activation <- 14
gen_ae_loss   <- 15


make_ae <- function(ind) {
  ae_f <- switch(
    ind[gen_ae_type],
    autoencoder,
    autoencoder_denoising,
    autoencoder_contractive,
    autoencoder_robust,
    autoencoder_sparse,
    autoencoder_variational
  )
  
  # Prepare layers' configuration
  idx <- 1:ind[gen_ae_layers]
  coder_lengths <- ind[rev(gen_ae_coding_length - idx)]
  decoder_lengths <- ind[gen_ae_coding_length - idx]
  coder_activations <-
    ae_layer_activation[ind[gen_ae_coding_activation - idx]]
  decoder_activations <-
    ae_layer_activation[ind[gen_ae_coding_activation + idx]]
  
  # Create the layers
  network <- input()
  if (ind[gen_ae_layers] == 0) {
    network <-
      network + dense(ind[gen_ae_coding_length], ae_layer_activation[ind[gen_ae_coding_activation]]) # encoding layer
  } else {
    for (idx in 1:ind[gen_ae_layers])
      network <-
        network + dense(coder_lengths[idx], coder_activations[idx])
    network <-
      network + dense(ind[gen_ae_coding_length], ae_layer_activation[ind[gen_ae_coding_activation]]) # encoding layer
    for (idx in 1:ind[gen_ae_layers])
      network <-
      network + dense(decoder_lengths[idx], decoder_activations[idx])
  }
  network <-
    network + output(ae_output_activation[ind[gen_ae_output_activation]])
  
  if (ind[gen_ae_type] == 4)
    return(ae_f(network))
  else
    return(ae_f(network, loss = ae_loss[ind[gen_ae_loss]]))
}

# cmake_ae <- cmpfun(make_ae, options = list(optimize = 3))

evaluate_mean_squared_error <- function(learner, data)  {
  k_model <- learner$models$autoencoder
  keras::compile(
    k_model,
    optimizer = "sgd",
    loss = ruta:::to_keras(learner$loss,
                           learner),
    metrics = keras::metric_mean_squared_error
  )
  keras::evaluate(k_model,
                  x = data,
                  y = data,
                  verbose = 0)[[2]]
}

main <- function(args) {
  options(keras.fit_verbose = 0)
  inputfile <- args[1]
  ind <- as.integer(strsplit(args[2], " ", fixed = T)[[1]])
  
  # Preparing the data
  load(inputfile)
  train_idx <- sample(1:nrow(dataset), size = nrow(dataset) * 0.8)
  dataset_train <- as.matrix(dataset[train_idx, ])
  dataset_test <- as.matrix(dataset[-train_idx, ])
  
  NF <-
    length(dataset)  # Number of features. It will be the number of units in the input and output layers
  
  learner <- purrr::quietly(make_ae)(ind)$result
  learner <- train(learner, dataset_train)
  result <- evaluate_mean_squared_error(learner, dataset_test)
  cat(result)
}

clargs <- commandArgs(trailingOnly = TRUE)
main(clargs)
