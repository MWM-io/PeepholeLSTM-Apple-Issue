//
//  ActivationTorchScript.swift
//  RNNBenchmark
//
//  Created by Pierre Cournut on 11/04/2022.
//

import Foundation

class ActivationTorchScript {
    
    // Init models
    private lazy var rnn_module: TorchModule = {
        if let filePath = Bundle.main.path(forResource: "parallel_birnn", ofType: ".pt"), let module =
            TorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError()
        }
    }()
    
    // Compute inference
    func computePredictions(nbRows: Int, nbColumns: Int, completion: @escaping (Result<ContiguousArray<Float>, Error>) -> Void) {
        enum PredictionError: Error {
            case predictionFailed
        }
        
        // Instantiate model input
        let inputSequence = Array(repeating: Array(repeating: 0.0 as NSNumber, count: nbColumns), count: nbRows)
        
        // Start prediction
        DispatchQueue.global().async {
            // Beat activation model inference
            guard let output = self.rnn_module.predictBeatActivation(preprocessed: inputSequence) else {
                completion(.failure(PredictionError.predictionFailed))
                return
            }
            
            // Fill output sequence
            var outputSequence: ContiguousArray<Float> = []
            var k = 0
            while k < output.count {
                outputSequence.append(Float(truncating: output[k]))
                k = k + 1
            }
            completion(.success(outputSequence))
        }
    }
    
}

