import * as tf from '@tensorflow/tfjs';
import { Tensor, sigmoid } from '@tensorflow/tfjs';
import fs from 'fs'



function stackSelfN(x: Tensor, n: number, axis: number = 0) {
    var stacks: Array<Tensor> = []
    //@ts-ignore
    for (var i = 0; i < n; i++) {
        stacks.push(x);
    }
    return tf.stack(stacks, axis);
}

function outterProduct(a: Tensor, b: Tensor) {

    var aStack = stackSelfN(a, b.shape[0]);

    var bStack = stackSelfN(b, a.shape[0], 1);


    return tf.mul(aStack, bStack);
}

export default class NeuralNet {

    w: Array<Tensor>;
    b: Array<Tensor>;

    constructor(layerSizes: Array<number>) {

        this.w = []
        this.b = []
        for (var i = 0; i < layerSizes.length - 1; i++) {
            this.w.push(tf.randomUniform([layerSizes[i + 1], layerSizes[i]], -1, 1))
            this.b.push(tf.randomUniform([layerSizes[i + 1]], -1, 1))
        }
    }

    d_sigmoid(x: Tensor) {
        var s = sigmoid(x)
        return tf.mul(s, tf.scalar(1).sub(s));
    }

    getValue(input: Tensor) {
        input = input.clone()
        var w = this.w
        var b = this.b
        for (var i = 0; i < w.length; i++) {
            input = tf.dot(w[i], input)
            input = tf.transpose(input)
            input = input.add(b[i])
            input = tf.transpose(input)
            input = tf.sigmoid(input)
        }

        return input
    }

    getValues(input: Tensor) {
        var w = this.w
        var b = this.b
        var zs = [input]
        var az = [input]
        for (var i = 0; i < w.length; i++) {
            var z = tf.dot(w[i], az[i])
            z = tf.transpose(z)
            z = z.add(b[i])
            z = tf.transpose(z)
            var a = tf.sigmoid(z)

            zs.push(z)
            az.push(a)
        }

        return [az, zs]
    }

    backPropagate(input: Tensor, output: Tensor, learning_rate: number = 1) {
        var w = this.w
        var b = this.b
        var [az, zs] = this.getValues(input)

        var dws: Array<Tensor> = [];
        var dbs: Array<Tensor> = [];

        var res = az[az.length - 1];

        var error = tf.mul(this.d_sigmoid(zs[zs.length - 1]), (tf.sub(res, output)))

        dbs.push(error)

        var a = az[az.length - 2]

        var dw = outterProduct(a, error);



        dws.push(dw)

        for (let i = w.length - 2; i >= 0; i--) {
            error = tf.mul(tf.dot(tf.transpose(w[i + 1]), error), this.d_sigmoid(zs[i + 1]))

            dw = outterProduct(az[i], error);

            dws.push(dw)
            dbs.push(error)
        }

        dws = dws.reverse();
        dbs = dbs.reverse();

        if (dws[0].shape.length === 3) {
            for (let i = 0; i < dws.length; i++) {
                w[i] = w[i].sub(dws[i].mean(2).mul(tf.scalar(learning_rate)))
            }

            for (let i = 0; i < dbs.length; i++) {
                b[i] = b[i].sub(dbs[i].mean(1).mul(tf.scalar(learning_rate)))
            }

        } else {

            for (let i = 0; i < dws.length; i++) {
                w[i] = w[i].sub(dws[i].mul(tf.scalar(learning_rate)))
            }


            for (let i = 0; i < dbs.length; i++) {
                b[i] = b[i].sub(dbs[i].mul(tf.scalar(learning_rate))).mul(tf.scalar(learning_rate))
            }

        }

    }

    oneHotHighestValue(input: Tensor) {
        var results = tf.transpose(this.getValue(input))
        if(input.shape[1])
            return tf.argMax(results, 1);
        return tf.argMax(results, 0);
    }

    accuracyOneHotHighestValue(input: Tensor, output: Tensor) {
        var results = this.oneHotHighestValue(input)
        
        var sames = output.equal(results).sum();
        //
        return sames.div(tf.scalar(output.shape[0]));
    }

    async toData() {
        return {
            w: await Promise.all(this.w.map(async w => ({ data: Object.values(await w.data()), shape: w.shape }))),
            b: await Promise.all(this.b.map(async b => ({ data: Object.values(await b.data()), shape: b.shape }))),
        }
    }

    async toJson() {
        return JSON.stringify(await this.toData())
    }

    async saveToFileLocal(name: string = "ai") {
        fs.writeFileSync(name+'.json', await this.toJson());
    }

    async loadFromFileLocal(name: string = "ai") {
        console.log("loading")
        try{
            var json = fs.readFileSync(name+'.json').toString();
            await this.loadFromJson(json);
        }catch(e){
          
        }
        
    }

    loadFromJson(json: string) {
        this.fromObject(JSON.parse(json))
    }

    fromObject({w,b}: any){
        this.w = w.map((w: any) => tf.tensor(w.data, w.shape));
        this.b = b.map((b: any) => tf.tensor(b.data, b.shape));
    }

}