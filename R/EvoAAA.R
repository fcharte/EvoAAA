#!/usr/bin/env Rscript
# Call: rscript evolutionaryAESearch.R EX|GA|EV|RND|DE dataset

dependencies <-
  c("compiler", "gramEvol", "wrapr", "ruta", "keras", "DEoptim")
to_be_installed <-
  !(dependencies %in% row.names(installed.packages()))
install.packages(dependencies[to_be_installed], repos = "https://cran.rstudio.com")

set.seed(73)

library(purrr)
library(compiler)
library(gramEvol)
library(DEoptim)

args = commandArgs(trailingOnly = TRUE)
inputfile <- paste0("../data/", args[2], ".rdata")
maxind <- as.numeric(args[3])

# Preparing the data
load(inputfile)

train_idx <- sample(1:nrow(dataset), size = nrow(dataset) * 0.8)
dataset_train <- as.matrix(dataset[train_idx, ])
dataset_test <- as.matrix(dataset[-train_idx, ])

NF <-
  length(dataset)  # Number of features. It will be the number of units in the input and output layers

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

# Create an AE from a chromosome

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

cmake_ae <- cmpfun(make_ae, options = list(optimize = 3))

# Set the GA/EV/DE configuration
genomeLen <- 15
genomeMin <- c(1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
genomeMax <- c(6, 3, NF, NF, NF, NF, 8, 8, 8, 8, 8, 8, 8, 4, 5)

costMin <- 0
costMax <- Inf
written_ind <- 0
evaluated_inds <<- 0
bestval <<- Inf
bestmem <<- genomeMin
last_result <<- bestval

suggested_initial <- matrix(
  c(
    2,
    0,
    NF %/% 2,
    NF %/% 3,
    NF %/% 4,
    NF %/% 5,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    1,
    1,
    4,
    0,
    NF %/% 2,
    NF %/% 3,
    NF %/% 4,
    NF %/% 5,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    1,
    1,
    2,
    2,
    NF %/% 2,
    NF %/% 3,
    NF %/% 4,
    NF %/% 5,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    1,
    1,
    4,
    2,
    NF %/% 2,
    NF %/% 3,
    NF %/% 4,
    NF %/% 5,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    1,
    1
  ),
  nrow = 4,
  byrow = TRUE
)

# Evaluation of individuals' fitness

complexity_penalization_factor = 1E-4 # Penalize the more complex configurations

evaluate.ind <- cmpfun(function(ind) {
  if (as.numeric(Sys.time()) - now > 86400)
    return (costMax)
  
  evaluated_inds <<- evaluated_inds + 1
  
  # Any invalid configuration will return maximum cost
  if (any(ind < genomeMin) |
      any(ind > genomeMax) |
      ind[3] < ind[4] |
      ind[4] < ind[5] |
      ind[5] < ind[6] |
      (ind[1] == 6 && ind[gen_ae_coding_length] == 1) |
      # softsign activation does not work yet with contractive autoencoders because
      # we need softsigngradgrad which is not defined in tensorflow
      (ind[gen_ae_type] == 3 &&
       8 %in% ind[7:gen_ae_output_activation]))
    return(costMax)
  
  cat(".")
  
  output <- system2(
    '/usr/bin/env',
    args = c(
      'Rscript',
      './eval_ae.R',
      inputfile,
      paste0('"', paste0(ind, collapse = " "), '"')
    ),
    stdout = TRUE,
    stderr = paste0(summary(outputfile)$description, ".log")
  )
  result <- as.numeric(output)
  
  timestamp <- as.numeric(Sys.time()) - now
  cat(timestamp, ",", result, "\n", sep = "", file = solutionsfile)
  
  value <-
    result + (ind[gen_ae_layers] + 1) * ind[gen_ae_coding_length] * complexity_penalization_factor
  last_result <<- value
  
  if (is.na(value) || is.nan(value))
    return(costMax)
  
  if ((args[1] == "DE" |
       args[1] == "ALPHA") &
      (evaluated_inds - written_ind >= 100 | last_result < bestval)) {
    written_ind <<- evaluated_inds
    if (last_result < bestval) {
      bestval <<- last_result
      bestmem <<- ind
    }
    cat(
      timestamp,
      ",",
      bestval,
      ",",
      paste(bestmem, collapse = ','),
      ",",
      evaluated_inds,
      "\n",
      sep = "",
      file = outputfile
    )
  }
  
  return(value)
}, options = list(optimize = 3))

# Function to monitor the GA/EV

last_reset <<- 0
reset_period <<- 10

monitor <- cmpfun(function(result) {
  timestamp <- as.numeric(Sys.time()) - now
  if (timestamp > 86400)
    return
  
  cat(
    timestamp,
    ",",
    result$best$cost,
    ",",
    paste(result$best$genome, collapse = ','),
    ",",
    evaluated_inds,
    "\n",
    sep = "",
    file = outputfile
  )
  
  if (evaluated_inds >= last_reset + reset_period) {
    last_reset <- evaluated_inds
    cat("-")
    gc()
  }
}, options = list(optimize = 3))

monitorDE <- cmpfun(function(result) {
  result <- round(result)
  
  result
}, options = list(optimize = 3))

header <-
  "Seconds,Fitness,GenAEType,GenAELayers,GenAELength1,GenAELength2,GenAELength3,GenAECodingLength,GenAEAct1,GenAEAct2,GenAEAct3,GenAEAct4,GenAEAct5,GenAEAct6,GenAEAct7,GenAEAct8,GenAELoss,EvaluatedInds\n"

if (length(args) == 0) {
  stop("Specify the strategy to run: RND, DE, GA, EV or EX", call. = FALSE)
}

if (args[1] == "GA") {
  # Run the GA
  
  outputfile <-
    file(paste0("GeneticAlg-", args[2], ".csv"), open = "w")
  solutionsfile <-
    file(paste0("GASolutions-", args[2], ".csv"), open = "w")
  cat(header, file = outputfile)
  
  evaluated_inds <<- 0
  now <- as.numeric(Sys.time())
  
  GA <- GeneticAlg.int(
    genomeLen = genomeLen,
    genomeMin = genomeMin,
    genomeMax = genomeMax,
    allowrepeat = TRUE,
    terminationCost = costMin,
    geneCrossoverPoints = c(1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1),
    suggestions = suggested_initial,
    monitorFunc = monitor,
    popSize = 25,
    evalFunc = evaluate.ind
  )
  
  cat("\n", evaluated_inds, "individuals evaluated\n", file = outputfile)
  cat(GA$population$currentIteration, "generations\n", file = outputfile)
  cat("Best fitness: ", GA$best$cost, "\n", file = outputfile)
  cat(
    "Best AE: c(",
    paste(GA$best$genome, collapse = ","),
    ")\n",
    sep = "",
    file = outputfile
  )
  complexity_penalization_factor = 0
  cat("Best AE loss:", evaluate.ind(GA$best$genome), "\n", file = outputfile)
  close(outputfile)
  close(solutionsfile)
  save(GA, file = paste0("GA-", args[2], ".RData"))
}

if (args[1] == "DE") {
  outputfile <-
    file(paste0("DifEvolution-", args[2], ".csv"), open = "w")
  solutionsfile <-
    file(paste0("DESolutions-", args[2], ".csv"), open = "w")
  cat(header, file = outputfile)
  
  evaluated_inds <<- 0
  bestval <<- Inf
  bestmem <<- genomeMin
  last_result <<- bestval
  written_ind <<- 0
  
  now <- as.numeric(Sys.time())
  
  DE <- DEoptim(
    fn = evaluate.ind,
    lower = genomeMin,
    upper = genomeMax,
    fnMap = monitorDE,
    control = DEoptim.control(
      VTR = costMin,
      itermax = 30,
      trace = TRUE
    )
  )
  
  cat("\n", evaluated_inds, "individuals evaluated\n", file = outputfile)
  cat(DE$optim$iter, "generations\n", file = outputfile)
  cat("Best fitness: ", bestval, "\n", file = outputfile)
  cat("Best AE: c(",
      paste(bestmem, collapse = ","),
      ")\n",
      sep = "",
      file = outputfile)
  complexity_penalization_factor = 0
  cat("Best AE loss:", evaluate.ind(bestmem), "\n", file = outputfile)
  close(outputfile)
  close(solutionsfile)
  save(DE, file = paste0("DE-", args[2], ".RData"))
}

if (args[1] == "ALPHA") {
  for (alpha in c(0.0001, 0.001, 0.01, 0.1, 1.0)) {
    complexity_penalization_factor = alpha
    
    outputfile <-
      file(paste0("DifEvolution-", args[2], "-", alpha, ".csv"), open = "w")
    solutionsfile <-
      file(paste0("DESolutions-", args[2], "-", alpha, ".csv"), open = "w")
    cat(header, file = outputfile)
    
    evaluated_inds <<- 0
    bestval <<- Inf
    bestmem <<- genomeMin
    last_result <<- bestval
    written_ind <<- 0
    
    now <- as.numeric(Sys.time())
    
    DE <- DEoptim(
      fn = evaluate.ind,
      lower = genomeMin,
      upper = genomeMax,
      fnMap = monitorDE,
      control = DEoptim.control(
        VTR = costMin,
        itermax = 30,
        trace = TRUE
      )
    )
    
    cat("\n", evaluated_inds, "individuals evaluated\n", file = outputfile)
    cat(DE$optim$iter, "generations\n", file = outputfile)
    cat("Best fitness: ", bestval, "\n", file = outputfile)
    cat(
      "Best AE: c(",
      paste(bestmem, collapse = ","),
      ")\n",
      sep = "",
      file = outputfile
    )
    bestval <-
      bestval - (bestmem[gen_ae_layers] + 1) * bestmem[gen_ae_coding_length] * complexity_penalization_factor
    cat("Best AE loss:", bestval, "\n", file = outputfile)
    close(outputfile)
    close(solutionsfile)
    save(DE, file = paste0("DE-", args[2], "-", alpha, ".RData"))
  }
}

if (args[1] == "EV") {
  # Run the EV
  outputfile <-
    file(paste0("EvolutionStrategy-", args[2], ".csv"), open = "w")
  solutionsfile <-
    file(paste0("EVSolutions-", args[2], ".csv"), open = "w")
  cat(header, file = outputfile)
  
  evaluated_inds <<- 0
  now <- as.numeric(Sys.time())
  
  EV <- EvolutionStrategy.int(
    genomeLen = genomeLen,
    genomeMin = genomeMin,
    genomeMax = genomeMax,
    terminationCost = costMin,
    monitorFunc = monitor,
    evalFunc = evaluate.ind
  )
  
  cat("\n", evaluated_inds, "individuals evaluated\n", file = outputfile)
  cat(EV$population$currentIteration, "generations\n", file = outputfile)
  cat("Best fitness: ", EV$best$cost, "\n", file = outputfile)
  cat(
    "Best AE: c(",
    paste(EV$best$genome, collapse = ","),
    ")\n",
    sep = "",
    file = outputfile
  )
  complexity_penalization_factor = 0
  cat("Best AE loss:", evaluate.ind(EV$best$genome), "\n", file = outputfile)
  close(outputfile)
  close(solutionsfile)
  save(EV, file = paste0("EV-", args[2], ".RData"))
}


if (args[1] == "EX") {
  # Perform an exhaustive search to find the best AE configuration
  library("wrapr")
  
  combinations_seen <- 0
  evaluated_inds <<- 0
  stop = FALSE
  stopTime <- Sys.time() + 24 * 60 * 60
  now <- as.numeric(Sys.time())
  
  outputfile <-
    file(paste0("Exhaustive-", args[2], ".csv"), open = "w")
  solutionsfile <-
    file(paste0("EXSolutions-", args[2], ".csv"), open = "w")
  cat(header, file = outputfile)
  
  best_ind <- genomeMin
  best_cost <- evaluate.ind(best_ind)
  
  for (AEType in seqi(genomeMin[1], genomeMax[1])) {
    for (AELayers in seqi(genomeMin[2], genomeMax[2])) {
      for (AEUnits1 in seqi(genomeMin[3], genomeMax[3])) {
        for (AEUnits2 in seqi(genomeMin[4], genomeMax[4])) {
          for (AEUnits3 in seqi(genomeMin[5], genomeMax[5])) {
            for (AEUnitsEncoding in seqi(genomeMin[6], genomeMax[6])) {
              for (AEAct1 in seqi(genomeMin[7], genomeMax[7])) {
                for (AEAct2 in seqi(genomeMin[8], genomeMax[8])) {
                  for (AEAct3 in seqi(genomeMin[9], genomeMax[9])) {
                    for (AEAct4 in seqi(genomeMin[10], genomeMax[10])) {
                      for (AEAct5 in seqi(genomeMin[11], genomeMax[11])) {
                        for (AEAct6 in seqi(genomeMin[12], genomeMax[12])) {
                          for (AEAct7 in seqi(genomeMin[13], genomeMax[13])) {
                            for (AEAct8 in seqi(genomeMin[14], genomeMax[14])) {
                              for (AELoss in seqi(genomeMin[15], genomeMax[15])) {
                                ind <-
                                  c(
                                    AEType,
                                    AELayers,
                                    AEUnits1,
                                    AEUnits2,
                                    AEUnits3,
                                    AEUnitsEncoding,
                                    AEAct1,
                                    AEAct2,
                                    AEAct3,
                                    AEAct4,
                                    AEAct5,
                                    AEAct6,
                                    AEAct7,
                                    AEAct8,
                                    AELoss
                                  )
                                fitness <- evaluate.ind(ind)
                                if (fitness < best_cost) {
                                  best_cost <- fitness
                                  best_ind <- ind
                                  monitor(
                                    list(
                                      combinations = combinations_seen,
                                      best = list(
                                        genome = best_ind,
                                        cost = best_cost
                                      )
                                    )
                                  )
                                }
                                combinations_seen <-
                                  combinations_seen + 1
                                if (stopTime < Sys.time())
                                  stop = TRUE
                                else if (combinations_seen %% 100 == 0) {
                                  monitor(list(
                                    best = list(
                                      genome = best_ind,
                                      cost = best_cost
                                    )
                                  ))
                                }
                                if (stop)
                                  break
                              }
                              if (stop)
                                break
                            }
                            if (stop)
                              break
                          }
                          if (stop)
                            break
                        }
                        if (stop)
                          break
                      }
                      if (stop)
                        break
                    }
                    if (stop)
                      break
                  }
                  if (stop)
                    break
                }
                if (stop)
                  break
              }
              if (stop)
                break
            }
            if (stop)
              break
          }
          if (stop)
            break
        }
        if (stop)
          break
      }
      if (stop)
        break
    }
    if (stop)
      break
  }
  cat("\n", evaluated_inds, "individuals evaluated\n", file = outputfile)
  cat("Best fitness: ", best_cost, "\n", file = outputfile)
  cat(
    "Best AE: c(",
    paste(best_ind, collapse = ","),
    ")\n",
    sep = "",
    file = outputfile
  )
  complexity_penalization_factor = 0
  cat("Best AE loss:", evaluate.ind(best_ind), "\n", file = outputfile)
  close(outputfile)
  close(solutionsfile)
}

if (args[1] == "RND") {
  # Perform an random search to find the best AE configuration
  library("wrapr")
  
  combinations_seen <- 0
  evaluated_inds <<- 0
  stop = FALSE
  stopTime <- Sys.time() + 24 * 60 * 60
  now <- as.numeric(Sys.time())
  
  outputfile <- file(paste0("Random-", args[2], ".csv"), open = "w")
  solutionsfile <-
    file(paste0("RNDSolutions-", args[2], ".csv"), open = "w")
  cat(header, file = outputfile)
  
  best_ind <- genomeMin
  best_cost <- evaluate.ind(best_ind)
  
  stop = FALSE
  while (!stop) {
    AEType    <- sample(genomeMin[1]:genomeMax[1], 1)
    AELayers  <- sample(genomeMin[1]:genomeMax[2], 1)
    AEUnits1  <- sample(genomeMin[3]:genomeMax[3], 1)
    AEUnits2  <- sample(genomeMin[4]:genomeMax[4], 1)
    AEUnits3  <- sample(genomeMin[5]:genomeMax[5], 1)
    AEUnitsEncoding <- sample(genomeMin[6]:genomeMax[6], 1)
    AEAct1    <- sample(genomeMin[7]:genomeMax[7], 1)
    AEAct2    <- sample(genomeMin[8]:genomeMax[7], 1)
    AEAct3    <- sample(genomeMin[9]:genomeMax[9], 1)
    AEAct4    <- sample(genomeMin[10]:genomeMax[10], 1)
    AEAct5    <- sample(genomeMin[11]:genomeMax[11], 1)
    AEAct6    <- sample(genomeMin[12]:genomeMax[12], 1)
    AEAct7    <- sample(genomeMin[13]:genomeMax[13], 1)
    AEAct8    <- sample(genomeMin[14]:genomeMax[14], 1)
    AELoss    <- sample(genomeMin[15]:genomeMax[15], 1)
    
    ind <-
      c(
        AEType,
        AELayers,
        AEUnits1,
        AEUnits2,
        AEUnits3,
        AEUnitsEncoding,
        AEAct1,
        AEAct2,
        AEAct3,
        AEAct4,
        AEAct5,
        AEAct6,
        AEAct7,
        AEAct8,
        AELoss
      )
    fitness <- evaluate.ind(ind)
    if (fitness < best_cost) {
      best_cost <- fitness
      best_ind <- ind
      monitor(list(
        combinations = combinations_seen,
        best = list(genome = best_ind, cost = best_cost)
      ))
    }
    combinations_seen <- combinations_seen + 1
    if (stopTime < Sys.time() | evaluated_inds >= maxind)
      stop = TRUE
    else if (combinations_seen %% 100 == 0) {
      monitor(list(best = list(
        genome = best_ind, cost = best_cost
      )))
    }
  }
  
  cat("\n", evaluated_inds, "individuals evaluated\n", file = outputfile)
  cat("Best fitness: ", best_cost, "\n", file = outputfile)
  cat(
    "Best AE: c(",
    paste(best_ind, collapse = ","),
    ")\n",
    sep = "",
    file = outputfile
  )
  complexity_penalization_factor = 0
  cat("Best AE loss:", evaluate.ind(best_ind), "\n", file = outputfile)
  close(outputfile)
  close(solutionsfile)
}
