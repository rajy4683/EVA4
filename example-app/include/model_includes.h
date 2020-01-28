#include <torch/torch.h>
#include <iostream>

struct TestNet : torch::nn::Module {
  TestNet(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};

struct EVA4BasicConv: torch::nn::Module {
  EVA4BasicConv():
        conv1(torch::nn::Conv2dOptions(1, 32, 3).padding(1).bias(false)),     
        conv2(torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)),
        pool1(torch::nn::MaxPool2dOptions(2).stride(2)),    
        conv3(torch::nn::Conv2dOptions(64, 128, 3).padding(1).bias(false)),  
        conv4(torch::nn::Conv2dOptions(128, 256, 3).padding(1).bias(false)),
        pool2(torch::nn::MaxPool2dOptions(2).stride(2)), 
        conv5(torch::nn::Conv2dOptions(256, 512, 3).padding(0).bias(false)),
        conv6(torch::nn::Conv2dOptions(512, 1024, 3).padding(0).bias(false)),
        conv7(torch::nn::Conv2dOptions(1024, 10, 3).padding(0).bias(false))

    {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("conv5", conv5);
        register_module("conv6", conv6);
        register_module("conv7", conv7);
        register_module("pool1", pool1);
        register_module("pool2", pool2);
    }
    torch::Tensor forward(torch::Tensor x) {
        x = pool1(torch::relu(conv2->forward(torch::relu(conv1->forward(x)))));
        x = pool2(torch::relu(conv4->forward(torch::relu(conv3->forward(x)))));
        x = torch::relu(conv6->forward(torch::relu(conv5->forward(x))));
        x = torch::relu(conv7->forward(x));
        x = x.view({-1,10});
        x = torch::nn::functional::detail::log_softmax(x, 1, c10::nullopt);
        //x = torch::log_softmax(x, 10);
        return x;
     }
    torch::nn::Conv2d conv1, conv2, conv3, conv4,conv5, conv6, conv7;
    torch::nn::MaxPool2d pool1, pool2;
 //nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};

//############## const globals #################
// Mostly taken from https://github.com/pytorch/examples/tree/master/cpp/mnist
// The batch size for testing and training.
const int64_t kTestBatchSize = 1000;
const int64_t kTrainBatchSize = 128;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10; 

// Path where MNIST dataset will be downloaded.
// ########## EDIT THIS PATH AS PER your directory structure or location where 
// ########## YOU WANT TO STORE THE MNIST dataset
const char* kDataFolder = "/content/gdrive/My Drive/cpp_dl/MNIST/raw";