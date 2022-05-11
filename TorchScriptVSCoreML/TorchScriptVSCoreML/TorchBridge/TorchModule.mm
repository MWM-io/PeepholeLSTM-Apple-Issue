//
//  TorchModule.m
//  TorchScriptVSCoreML
//
//  Created by Pierre Cournut on 10/05/2022.
//

#import <Foundation/Foundation.h>
#import "TorchModule.h"
#import <Libtorch/Libtorch.h>

@implementation TorchModule {
 @protected
  torch::jit::script::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
  self = [super init];
  if (self) {
    try {
      _impl = torch::jit::load(filePath.UTF8String);
      _impl.eval();
    } catch (const std::exception& exception) {
      NSLog(@"%s", exception.what());
      return nil;
    }
  }
  return self;
}

- (NSArray<NSNumber*>*)predictBeatActivation:(NSArray<NSArray<NSNumber*>*>*)preprocessedAudio {
    try {
        
        // input
        int nbFrames = preprocessedAudio.count;
        int nbDim = preprocessedAudio[0].count;
        std::vector<std::vector<Float32>> vect_2d(nbFrames, std::vector<Float32>(nbDim, 0));
        for (int i = 0; i < nbFrames; i++) {
            std::vector<Float32> vect_1d(nbDim, 0);
            for (int j = 0; j < nbDim; j++) {
                vect_1d[j] = [preprocessedAudio[i][j] doubleValue];
            }
            vect_2d[i] = vect_1d;
        }

        // copy vector values into tensor
        auto options = torch::TensorOptions().dtype(at::kFloat);
        auto tensor = torch::zeros({nbFrames, nbDim}, options);
        for (int i = 0; i < nbFrames; i++) {
            tensor.slice(0, i, i+1) = torch::from_blob(vect_2d[i].data(), {nbDim}, options);
        }
        
        // run inference
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        at::Tensor outputTensor = _impl.forward({tensor}).toTensor();
        
        // format and display result
        int tensorLength = int(outputTensor.size(0));
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            std::cout << "got nothing";
            return nil;
        }
        NSMutableArray* results = [[NSMutableArray alloc] init];
        
        // add output from tensor to array
        for (int i = 0; i < tensorLength; i++) {
            [results addObject:@(floatBuffer[i])];
        }
        
        // return array of format NSArray<NSNumber>
        return [results copy];
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end
