
import * as tf from '@tensorflow/tfjs';
import NeuralNetwork from './NeuralNet';


//@ts-ignore
import MnistData from './data';

export default async function test() {
    console.log("Starting")
    var mnist = new MnistData();
    
    //@ts-ignore
    var { xs, labels, labelsi } = (await mnist.getTestDataLocal(501,10))[0]; //mnist.getTestData(2);


    //@ts-ignore
    var nn = new NeuralNetwork([xs.shape[1], 64, 32, labels.shape[1]])
    await nn.loadFromFileLocal("src/ai_64_32_3")


    var xs_T = tf.transpose(xs);
    var labels_T = tf.transpose(labels);

    labelsi.print()
    nn.oneHotHighestValue(xs_T).print();
    console.log("accuracy" + nn.accuracyOneHotHighestValue(xs_T, labelsi)) 



}

test();