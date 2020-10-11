/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

export const IMAGE_H = 28;
export const IMAGE_W = 28;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';


function shuffle<T>(array : Array<T>) {
  let counter = array.length;

  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    let index = Math.floor(Math.random() * counter);

    // Decrease counter by 1
    counter--;

    // And swap the last element with it
    let temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }

  return array;
}


// interface MNIST_DATA {
//   label: String,
//   image: number[]
// }

/**
 * A class that fetches the sprited MNIST dataset and provide data as
 * tf.Tensors.
 */

export default class MnistData {
  constructor() { }

  // async load() {
  //   // Make a request for the MNIST sprited image.
  //   const img = new Image();
  //   const canvas = document.createElement('canvas');
  //   const ctx = canvas.getContext('2d');
  //   const imgRequest = new Promise((resolve, reject) => {
  //     img.crossOrigin = '';
  //     img.onload = () => {
  //       img.width = img.naturalWidth;
  //       img.height = img.naturalHeight;

  //       const datasetBytesBuffer =
  //         new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

  //       const chunkSize = 5000;
  //       canvas.width = img.width;
  //       canvas.height = chunkSize;

  //       for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
  //         const datasetBytesView = new Float32Array(
  //           datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
  //           IMAGE_SIZE * chunkSize);
  //         ctx.drawImage(
  //           img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
  //           chunkSize);

  //         const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  //         for (let j = 0; j < imageData.data.length / 4; j++) {
  //           // All channels hold an equal value since the image is grayscale, so
  //           // just read the red channel.
  //           datasetBytesView[j] = imageData.data[j * 4] / 255;
  //         }
  //       }
  //       this.datasetImages = new Float32Array(datasetBytesBuffer);

  //       resolve();
  //     };
  //     img.src = MNIST_IMAGES_SPRITE_PATH;
  //   });

  //   const labelsRequest = fetch(MNIST_LABELS_PATH);
  //   const [imgResponse, labelsResponse] =
  //     await Promise.all([imgRequest, labelsRequest]);

  //   this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

  //   // Slice the the images and labels into train and test sets.
  //   this.trainImages =
  //     this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
  //   this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
  //   this.trainLabels =
  //     this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  //   this.testLabels =
  //     this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  // }


  async getTestDataLocal(offset: number, amount: number, groups = 1, junkPercent?:number) {
    var mnist_data: Array<any> = require("./mnist_handwritten_test.json");
    console.log(mnist_data.length)
    mnist_data = mnist_data.slice(offset, offset + amount * groups);
    if(junkPercent){
        var junkCount = Math.floor(mnist_data.length * junkPercent);
        for(var i = 0; i < junkCount; i++){
          var image = []
          for(var j = 0; j < 28*28; j++){
            image.push(Math.random()%256);
          }
          mnist_data.push({label: 10, image})
        }

    }
    shuffle(mnist_data);

    var trainingSets: Array<any> = []
    for (var i = 0; i < groups; i++) {
      var data = mnist_data.slice(amount * i, amount * i + amount)
      trainingSets.push(this.getGroupFromData(data, junkPercent?11:10))
    }
    return trainingSets
  }

  getGroupFromData(mnist_data: Array<any>, outputs:number) {
    var imgData = mnist_data.map(d => d.image);
    var size = imgData[0].length;
    var xs = tf.tensor(imgData, [imgData.length, size]);
    xs = tf.div(xs, tf.scalar(255));

    var labelData = mnist_data.map(d => d.label);
    var labelsi = tf.tensor1d(labelData, "int32");
    var labels = tf.oneHot(labelsi, outputs);
    return { xs, labels, labelsi }
  }

  /**
   * Get all training data as a data tensor and a labels tensor.
   *
   * @returns
   *   xs: The data tensor, of shape `[numTrainExamples, 28, 28, 1]`.
   *   labels: The one-hot encoded labels tensor, of shape
   *     `[numTrainExamples, 10]`.
   */
  // getTrainData() {
  //   const xs = tf.tensor4d(
  //     this.trainImages,
  //     [this.trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
  //   const labels = tf.tensor2d(
  //     this.trainLabels, [this.trainLabels.length / NUM_CLASSES, NUM_CLASSES]);
  //   return { xs, labels };
  // }

  /**
   * Get all test data as a data tensor a a labels tensor.
   *
   * @param {number} numExamples Optional number of examples to get. If not
   *     provided,
   *   all test examples will be returned.
   * @returns
   *   xs: The data tensor, of shape `[numTestExamples, 28, 28, 1]`.
   *   labels: The one-hot encoded labels tensor, of shape
   *     `[numTestExamples, 10]`.
   */
  // getTestData(numExamples) {
  //   let xs = tf.tensor2d(
  //     this.testImages,
  //     [this.testImages.length / IMAGE_SIZE, IMAGE_H * IMAGE_W]);
  //   let labels = tf.tensor2d(
  //     this.testLabels, [this.testLabels.length / NUM_CLASSES, NUM_CLASSES]);

  //   if (numExamples != null) {
  //     xs = xs.slice([0, 0], [numExamples, IMAGE_H * IMAGE_W]);
  //     labels = labels.slice([0, 0], [numExamples, NUM_CLASSES]);
  //   }
  //   return { xs, labels };
  // }
}