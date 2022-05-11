//
//  ContentView.swift
//  TorchScriptVSCoreML
//
//  Created by Pierre Cournut on 10/05/2022.
//

import SwiftUI

struct ContentView: View {
    
    @ObservedObject var contentViewModel = ContentViewModel()
    
    @State var sequenceLength: Double = 3000
    
    var body: some View {
        
        Slider(value: $sequenceLength, in: 100...10000, step: 100)
        Text("\(String(Int(sequenceLength))) timesteps in input sequence")
            .font(.title3)
        
        Text("CoreML")
            .font(.title2)
            .padding()
            .frame(minWidth: 0, maxWidth: 350)
            .background(RoundedRectangle(cornerRadius: 15).fill(.blue))
            .onTapGesture(perform: {
                contentViewModel.computeCoreMLPredictions(timesteps: Int(sequenceLength))
            })
        
        Text("TorchScript")
            .font(.title2)
            .padding()
            .frame(minWidth: 0, maxWidth: 350)
            .background(RoundedRectangle(cornerRadius: 15).fill(.green))
            .onTapGesture(perform: {
                contentViewModel.computeTorchScriptPredictions(timesteps: Int(sequenceLength))
            })

    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
