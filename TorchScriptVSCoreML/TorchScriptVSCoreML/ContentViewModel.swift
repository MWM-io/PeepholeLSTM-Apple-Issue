//
//  ContentViewModel.swift
//  RNNBenchmark
//
//  Created by Pierre Cournut on 11/04/2022.
//

import Foundation

class ContentViewModel: ObservableObject {
    
    var activationCoreML = ActivationCoreML()
    var activationTorchScript = ActivationTorchScript()
    
    func computeCoreMLPredictions(timesteps: Int) {
        let start = DispatchTime.now()
        activationCoreML.computePredictions(nbRows: timesteps, nbColumns: 314) { result in
            switch result {
            case .success(_):
                let computeTime = Float(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
                print("CoreML compute time: \(computeTime)")
            case .failure(let error):
                print("Error: \(error.localizedDescription)")
            }
        }
    }
    
    
    func computeTorchScriptPredictions(timesteps: Int) {
        let start = DispatchTime.now()
        activationTorchScript.computePredictions(nbRows: timesteps, nbColumns: 314) { result in
            switch result {
            case .success(_):
                let computeTime = Float(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
                print("TorchScript compute time: \(computeTime)")
            case .failure(let error):
                print("Error: \(error.localizedDescription)")
            }
        }
    }
    
}
