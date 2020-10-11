
import * as tf from '@tensorflow/tfjs';
import NeuralNetwork from './NeuralNet';


//@ts-ignore
import MnistData from './data';
import { math } from '@tensorflow/tfjs';
import test from './test';

async function train() {
    console.log("Starting")
    var mnist = new MnistData();
    var groups = 10;
    var total = 10000
    var sessionSize = 50;

    var name = "src/ai_64_32_1_junk";
    //@ts-ignore
    var nn = new NeuralNetwork([28 * 28, 64, 32, 11])
    await nn.loadFromFileLocal(name)



    for (let i = 0; i < 100; i++) {
        console.log("Next Iteration:")
        var trainingSets = await mnist.getTestDataLocal(0, sessionSize, total/sessionSize, 0.05); //mnist.getTestData(2);
        console.log("Finished Loading:")
        for (let k = 0; k < groups; k++) {
            var { xs, labels, labelsi } = trainingSets[k];
            var xs_T = tf.transpose(xs);
            var labels_T = tf.transpose(labels);

            for (let j = 0; j < 10; j++) {
                nn.backPropagate(xs_T, labels_T, 1);

            }
            var r = nn.getValue(xs_T)
            tf.sum(tf.square(tf.mean(tf.sub(r, labels_T)))).print()
            labelsi.print()
            nn.oneHotHighestValue(xs_T).print();
            console.log("accuracy" + nn.accuracyOneHotHighestValue(xs_T, labelsi))

            
        }
        console.log("Saving")
        await nn.saveToFileLocal(name);
        console.log("Saved")
    }

    await nn.saveToFileLocal(name)


    test();



}

train();