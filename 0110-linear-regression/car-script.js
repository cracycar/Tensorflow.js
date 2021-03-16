let inputs_max;
let labels_max;

//carsData data 중 자동차 마력을 기준으로 연비 예측하는 Linear Model 작성
//Data load
async function getData(){
    const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json'); 
    const carsData = await carsDataReq.json();
    const data = carsData.map(x => ({
        mpg: x.Miles_per_Gallon,
        hp:  x.Horsepower,
    })).filter(x => (x.mpg != null && x.hp != null));
    // data 를 tensor 로 convert 하고 shuffling, normalizingthe 
    return tf.tidy(()=>{
        tf.util.shuffle(data)

        let inputs = data.map(x => x.hp);
        let labels = data.map(x => x.mpg);

        inputs = tf.tensor2d(inputs, [inputs.length, 1]);
        inputs_max = inputs.max().dataSync();
        inputs = inputs.div(inputs_max);

        labels = tf.tensor2d(labels, [labels.length, 1]);
        labels_max = labels.max().dataSync();
        labels = labels.div(labels_max);

        return {
            inputs: inputs,
            labels: labels
        }
    });
}

// NN 모델 생성
function createModel(){
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
    model.add(tf.layers.dense({units: 1, useBias: true}));
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'meanSquaredError',
        metrics: ['mse']
    })
    return model;
}
// NN model compile and train
async function trainModel(model, inputs, labels){
    const batchSize = 32;
    const epochs = 100; // loop 몇 번 Train 할 것인지
    await model.fit(inputs, labels, {
        batchSize, epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            {name: 'Training Performance'},
            ['mse'],
            {height: 200, callbacks: ['onEpochEnd']}
        )
    });
}

function testModel(model, inputs, labels){
    let preds = model.predict(inputs).dataSync();
    labels = labels.dataSync();

    //시각화 하여 원래 input 의 label 과 prediction 비교
    const series1 = [];
    const series2 = [];
    for (let i=0; i < labels.length; i++){
        series1.push({x: i* inputs_max[0], y: preds[i] * labels_max[0]});
        series2.push({x: i* inputs_max[0], y: labels[i] * labels_max[0]});
    }
    tfvis.render.scatterplot(
        {name: 'Preds vs Original'},
        {values: [series1, series2], series: ['Preds', 'Original']},
        {xLabel: '마력', yLabel: '연비', height: 200}
    )
}

async function run(){
    const data = await getData();
    //Train/Test data split
    const train_size = Math.floor(data.inputs.shape[0] * 0.9);
    const test_size = data.inputs.shape[0] - train_size;
    const [X_train, X_test] = data.inputs.split([train_size, test_size]);
    const [y_train, y_test] = data.labels.split([train_size, test_size]);
    //model 생성
    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);
    //model train
    await trainModel(model, X_train, y_train);
    //model validatation
    testModel(model, X_test, y_test);
    console.log('Complete !!')
}

document.addEventListener('DOMContentLoaded', run);
