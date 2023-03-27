load('split_data_matrix.Rdata')
train_labels_cat <- to_categorical(train_labels, 2)

FLAGS <- flags(flag_numeric("nodes_layer1", 128),
               flag_numeric("nodes_layer2", 64),
               flag_numeric("nodes_layer3", 32),
               flag_numeric("dropout_layer1", 0.2),
               flag_numeric("dropout_layer2", 0.2),
               flag_numeric("dropout_layer3", 0.2))

model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes_layer1, activation = "relu", input_shape = ncol(train_data)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout_layer1) %>%
  layer_dense(units = FLAGS$nodes_layer2, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout_layer2) %>%
  layer_dense(units = FLAGS$nodes_layer3, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout_layer3) %>%
  layer_dense(units = 2, activation = "sigmoid") %>%
  compile(loss = "binary_crossentropy",
          optimizer = optimizer_adam(), 
          metrics = "AUC") %>% 
  fit(x = train_data, 
      y = train_labels_cat, 
      epochs = 50, 
      batch_size = 256,
      validation_split = 0.2,
      class_weight = list('0'=1,'1'=30),
      callbacks = list(callback_early_stopping(patience = 10),
                       callback_reduce_lr_on_plateau()),
      verbose = 2)

