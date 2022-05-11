//
//  TorchModule.h
//  TorchScriptVSCoreML
//
//  Created by Pierre Cournut on 10/05/2022.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchModule : NSObject

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
    NS_SWIFT_NAME(init(fileAtPath:))NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (nullable NSArray<NSNumber*>*)predictBeatActivation: (NSArray<NSArray<NSNumber*>*>*)preprocessedAudio
__attribute__((swift_name("predictBeatActivation(preprocessed:)")));

@end

NS_ASSUME_NONNULL_END
