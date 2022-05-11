//
//  ActivationCoreML.swift
//  RNNBenchmark
//
//  Created by Pierre Cournut on 11/04/2022.
//

import CoreML
import Foundation

class ActivationCoreML {
    
    typealias MLModel = birnn
    var model: MLModel!
    
    public init() {
        let config = MLModelConfiguration()
        model = try! MLModel(configuration: config)
    }

    func computePredictions(nbRows: Int, nbColumns: Int, completion: @escaping (Result<ContiguousArray<Float>, Error>) -> Void) {
        enum PredictionError: Error {
            case predictionFailed
        }
        
        DispatchQueue.global().async {
            do {
                // Cast Array to MLMultiArray
                let shape = [1, nbRows as NSNumber, nbColumns as NSNumber]
                let mlMelSpec = try MLMultiArray(shape: shape, dataType: MLMultiArrayDataType.float32)
                
                // Prediction
                let activations = try! self.model.prediction(melspec: mlMelSpec).activations
                
                // Cast back to ContiguousArray
                var result = ContiguousArray(repeating: Float(0.0), count: activations.count)
                for j in 0..<activations.count {
                    result[j] = Float(truncating: activations[j])
                }
                completion(.success(result))
            } catch {
                completion(.failure(PredictionError.predictionFailed))
            }
        }
    }
    
}
