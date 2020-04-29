library(readr)
library(dplyr)
library(ggplot2)
library(umap)

################################################################################

# load the original MNIST train and test data
# https://pjreddie.com/media/files/mnist_train.csv
# https://pjreddie.com/media/files/mnist_test.csv
mnist_train <- readr::read_csv("data-raw/mnist_train.csv", col_names = FALSE)
mnist_test <- readr::read_csv("data-raw/mnist_test.csv", col_names = FALSE)
names(mnist_train) <- c("label", paste0("pixel", 1:784))
names(mnist_test) <- c("label", paste0("pixel", 1:784))

# load the Fashion MNIST test data 
# https://www.kaggle.com/zalando-research/fashionmnist
fmnist_train <- readr::read_csv("data-raw/fashion-mnist_train.csv")
fmnist_test <- readr::read_csv("data-raw/fashion-mnist_test.csv")

mnist_train[-1] <- mnist_train[-1] / 255
fmnist_train[-1] <- fmnist_train[-1] / 255
fmnist_test[-1] <- fmnist_test[-1] / 255

if(!all.equal(dim(mnist_train), c(60000, 785))) stop("mnist_train has wrong dims.")
if(!all.equal(dim(mnist_test), c(10000, 785))) stop("mnist_test has wrong dims.")
if(!all.equal(dim(fmnist_train), c(60000, 785))) stop("fmnist_train has wrong dims.")
if(!all.equal(dim(fmnist_test), c(10000, 785))) stop("fmnist_test has wrong dims.")

################################################################################

set.seed(392663)

# get a UMAP embedding based on MNIST and based on FMNIST
mnist_umap_fit <- umap::umap(as.matrix(mnist_train[-1]))
fmnist_umap_fit <- umap::umap(as.matrix(fmnist_train[-1]))

save(mnist_umap_fit, file = "data/mnist_umap_fit.rda")
save(fmnist_umap_fit, file = "data/fmnist_umap_fit.rda")

################################################################################

# project the Fashion MNIST data onto the UMAP embedding trained on MNIST data
mnist_umap_fmnist_train <- predict(mnist_umap_fit, fmnist_train[-1])
mnist_umap_fmnist_test <- predict(mnist_umap_fit, fmnist_test[-1])
mnist_umap_mnist_test <- predict(mnist_umap_fit, mnist_test[-1])

mnist_umap_fmnist_train <- data.frame(mnist_umap_fmnist_train)
names(mnist_umap_fmnist_train) <- c("x1", "x2")
mnist_umap_fmnist_train$label <- fmnist_train$label

mnist_umap_fmnist_test <- data.frame(mnist_umap_fmnist_test)
names(mnist_umap_fmnist_test) <- c("x1", "x2")
mnist_umap_fmnist_test$label <- fmnist_test$label

mnist_umap_mnist_train <- data.frame(mnist_umap_fit$layout)
names(mnist_umap_mnist_train) <- c("x1", "x2")
mnist_umap_mnist_train$label <- mnist_train$label

mnist_umap_mnist_test <- data.frame(mnist_umap_mnist_test)
names(mnist_umap_mnist_test) <- c("x1", "x2")
mnist_umap_mnist_test$label <- mnist_test$label

readr::write_csv(mnist_umap_fmnist_train, "data/mnist_umap_fmnist_train.csv")
readr::write_csv(mnist_umap_fmnist_test, "data/mnist_umap_fmnist_test.csv")
readr::write_csv(mnist_umap_mnist_train, "data/mnist_umap_mnist_train.csv")
readr::write_csv(mnist_umap_mnist_test, "data/mnist_umap_mnist_test.csv")

# project the MNIST data onto the UMAP embedding trained on FMNIST data

fmnist_umap_mnist_train <- predict(fmnist_umap_fit, mnist_train[-1])
fmnist_umap_mnist_test <- predict(fmnist_umap_fit, mnist_test[-1])
fmnist_umap_fmnist_test <- predict(fmnist_umap_fit, fmnist_test[-1])

fmnist_umap_mnist_train <- data.frame(fmnist_umap_mnist_train)
names(fmnist_umap_mnist_train) <- c("x1", "x2")
fmnist_umap_mnist_train$label <- mnist_train$label

fmnist_umap_mnist_test <- data.frame(fmnist_umap_mnist_test)
names(fmnist_umap_mnist_test) <- c("x1", "x2")
fmnist_umap_mnist_test$label <- mnist_test$label

fmnist_umap_fmnist_train <- data.frame(fmnist_umap_fit$layout)
names(fmnist_umap_fmnist_train) <- c("x1", "x2")
fmnist_umap_fmnist_train$label <- fmnist_train$label

fmnist_umap_fmnist_test <- data.frame(fmnist_umap_fmnist_test)
names(fmnist_umap_fmnist_test) <- c("x1", "x2")
fmnist_umap_fmnist_test$label <- fmnist_test$label

readr::write_csv(fmnist_umap_mnist_train, "data/fmnist_umap_mnist_train.csv")
readr::write_csv(fmnist_umap_mnist_test, "data/fmnist_umap_mnist_test.csv")
readr::write_csv(fmnist_umap_fmnist_train, "data/fmnist_umap_fmnist_train.csv")
readr::write_csv(fmnist_umap_fmnist_test, "data/fmnist_umap_fmnist_test.csv")

################################################################################

#' Sample observations to be used as few-shots
#' 
#' @param label The label of the current observation
#' @param n The number of observations with label different from the current
#'     label to be included in the set of few-shots (for "training")
#' @param data The data set from which to draw the observations
#' @return Dataframe with n rows and three columns: 
#'     two coordinates and the label
#' @examples 
#' # Five-shots where 1 example observation will have label 7
#' get_examples(7, 4, mnist_umap_fmnist_train)
get_examples <- function(label, n, data) {
  label_same <- sample_n(data[data$label == label,], 1)
  label_diff <- sample_n(data[data$label != label,], n)
  
  examples <- bind_rows(label_same, label_diff) %>%
    rename(x1_example = x1, x2_example = x2, label_example = label)
  
  return(examples)
}

#' Join examples against test set
#' 
#' @param test Dataframe with observations to be predicted; 
#'     with columns label, x1, x2
#' @param example_source Dataframe with observations to be used as training
#'     examples; with columns label, x1, x2
#' @param n The number of observations with label different from the current
#'     label to be included in the set of few-shots (for "training")
#' @return Dataframe with n*dim(test)[1] rows
#' @examples 
#' # Get 5 examples each for the first two rows of mnist_umap_fmnist_test
#' join_examples(mnist_umap_fmnist_test[1:2,], mnist_umap_fmnist_train, 4)
join_examples <- function(test, example_source, n, seed = NA) {
  if(!is.na(seed)) set.seed(seed)
  
  test %>%
    dplyr::mutate(test_index = 1:n()) %>%
    dplyr::group_by(test_index, label, x1, x2) %>%
    dplyr::mutate(example_df = purrr::map(label, get_examples, n = n,
                                          data = example_source)) %>%
    tidyr::unnest(example_df) %>%
    dplyr::ungroup()
}

################################################################################

# We now perform a couple of experiments. For different sample sizes of
# n+1 = {1+1, 4+1, 9+1}, we will predict the following combinations:

# 1) FMNIST Test using MNIST Embedding with FMNIST Train
# 2) MNIST Test using MNIST Embedding with MNIST Test
# 3) MNIST Test using FMNIST Embedding with MNIST Train
# 4) FMNIST Test using FMNIST Embedding with FMNIST Test

################################################################################

mnist_umap_fmnist_test_examples_2 <- join_examples(mnist_umap_fmnist_test,
                                                   mnist_umap_fmnist_train, 
                                                   n = 1, seed = 427)
readr::write_csv(mnist_umap_fmnist_test_examples_2,
                 "data/mnist_umap_fmnist_test_examples_2.csv")
mnist_umap_fmnist_test_examples_4 <- join_examples(mnist_umap_fmnist_test,
                                                   mnist_umap_fmnist_train, 
                                                   n = 4, seed = 359)
readr::write_csv(mnist_umap_fmnist_test_examples_4,
                 "data/mnist_umap_fmnist_test_examples_4.csv")
mnist_umap_fmnist_test_examples_9 <- join_examples(mnist_umap_fmnist_test,
                                                   mnist_umap_fmnist_train, 
                                                   n = 9, seed = 9852)
readr::write_csv(mnist_umap_fmnist_test_examples_9,
                 "data/mnist_umap_fmnist_test_examples_9.csv")

mnist_umap_mnist_test_examples_2 <- join_examples(mnist_umap_mnist_test,
                                                  mnist_umap_mnist_test,  
                                                  n = 1, seed = 22)
readr::write_csv(mnist_umap_mnist_test_examples_2,
                 "data/mnist_umap_mnist_test_examples_2.csv")
mnist_umap_mnist_test_examples_4 <- join_examples(mnist_umap_mnist_test,
                                                  mnist_umap_mnist_test,  
                                                  n = 4, seed = 773)
readr::write_csv(mnist_umap_mnist_test_examples_4,
                 "data/mnist_umap_mnist_test_examples_4.csv")
mnist_umap_mnist_test_examples_9 <- join_examples(mnist_umap_mnist_test,
                                                  mnist_umap_mnist_test, 
                                                  n = 9, seed = 833)
readr::write_csv(mnist_umap_mnist_test_examples_9,
                 "data/mnist_umap_mnist_test_examples_9.csv")

fmnist_umap_mnist_test_examples_2 <- join_examples(fmnist_umap_mnist_test,
                                                   fmnist_umap_mnist_train, 
                                                   n = 1, seed = 353)
readr::write_csv(fmnist_umap_mnist_test_examples_2,
                 "data/fmnist_umap_mnist_test_examples_2.csv")
fmnist_umap_mnist_test_examples_4 <- join_examples(fmnist_umap_mnist_test,
                                                   fmnist_umap_mnist_train,  
                                                   n = 4, seed = 555)
readr::write_csv(fmnist_umap_mnist_test_examples_4,
                 "data/fmnist_umap_mnist_test_examples_4.csv")
fmnist_umap_mnist_test_examples_9 <- join_examples(fmnist_umap_mnist_test,
                                                   fmnist_umap_mnist_train,  
                                                   n = 9, seed = 9483)
readr::write_csv(fmnist_umap_mnist_test_examples_9,
                 "data/fmnist_umap_mnist_test_examples_9.csv")

fmnist_umap_fmnist_test_examples_2 <- join_examples(fmnist_umap_fmnist_test,
                                                    fmnist_umap_fmnist_test, 
                                                    n = 1, seed = 7339)
readr::write_csv(fmnist_umap_fmnist_test_examples_2,
                 "data/fmnist_umap_fmnist_test_examples_2.csv")
fmnist_umap_fmnist_test_examples_4 <- join_examples(fmnist_umap_fmnist_test,
                                                    fmnist_umap_fmnist_test, 
                                                    n = 4, seed = 92545)
readr::write_csv(fmnist_umap_fmnist_test_examples_4,
                 "data/fmnist_umap_fmnist_test_examples_4.csv")
fmnist_umap_fmnist_test_examples_9 <- join_examples(fmnist_umap_fmnist_test,
                                                    fmnist_umap_fmnist_test,  
                                                    n = 9, seed = 1235)
readr::write_csv(fmnist_umap_fmnist_test_examples_9,
                 "data/fmnist_umap_fmnist_test_examples_9.csv")

################################################################################

# function that runs 1-nearest-neighbor classification using the 
# previously sampled examples, evaluates the accuracy and compares it against
# random classification baselines
evaluate_knn_pred <- function(test_examples) {
  
  nn_pred <- test_examples %>%
    dplyr::mutate(dist = sqrt(((x1 - x1_example)^2 + 
                                 (x2 - x2_example)^2) / 2)) %>%
    dplyr::group_by(test_index, label) %>%
    dplyr::summarize(label_pred = label_example[dist == min(dist)]) %>%
    dplyr::ungroup()
  
  nn_pred$label_pred_random <- sample(0:9, size = nrow(nn_pred), replace = TRUE)
  
  test_example_random <- test_examples %>%
    dplyr::group_by(test_index) %>%
    dplyr::sample_n(1) %>%
    dplyr::rename(label_example_random = label_example) %>%
    dplyr::ungroup() %>%
    dplyr::select(test_index, label_example_random)
  
  acc_per_class <- nn_pred %>%
    dplyr::inner_join(test_example_random) %>%
    dplyr::group_by(label) %>%
    dplyr::summarize(acc = mean(label == label_pred),
                     acc_r = mean(label == label_pred_random),
                     acc_er = mean(label == label_example_random))
  
  acc <- nn_pred %>%
    dplyr::inner_join(test_example_random) %>%
    dplyr::summarize(acc = mean(label == label_pred),
                     acc_r = mean(label == label_pred_random),
                     acc_er = mean(label == label_example_random))
  
  return(list(acc = acc, acc_per_class = acc_per_class))
}

evaluate_knn_pred(mnist_umap_fmnist_test_examples_2)
evaluate_knn_pred(mnist_umap_fmnist_test_examples_4)
evaluate_knn_pred(mnist_umap_fmnist_test_examples_9)

evaluate_knn_pred(mnist_umap_mnist_test_examples_2)
evaluate_knn_pred(mnist_umap_mnist_test_examples_4)
evaluate_knn_pred(mnist_umap_mnist_test_examples_9)

evaluate_knn_pred(fmnist_umap_mnist_test_examples_2)
evaluate_knn_pred(fmnist_umap_mnist_test_examples_4)
evaluate_knn_pred(fmnist_umap_mnist_test_examples_9)

evaluate_knn_pred(fmnist_umap_fmnist_test_examples_2)
evaluate_knn_pred(fmnist_umap_fmnist_test_examples_4)
evaluate_knn_pred(fmnist_umap_fmnist_test_examples_9)

################################################################################

