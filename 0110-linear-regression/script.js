let inputs_max;
let labels_max;

// data를 가져오는 함수
async function getData(){
    //data 가져오기
    const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataReq.json();


   // arr.map : 배열의 각 요소를 불러내 callback 함수의 반환값으로 새로운 배열을 만듦
    const data = carsData.map(x => ({
        mpg: x.Miles_per_Gallon,
        hp:  x.Horsepower,
    })).filter(x =>(x.mpg != null && x.hp != null)); 
/*
    const data = carsData.map(function(x){
        return {
            mpg: x.Miles_per_Gallon,
            hp: x.Horsepower,
        };
    }).filter(function(x){
        if(x.mpg != null && x.hp != null){
            return x;
        }
    });
*/
    return tf.tidy(() => {
        tf.util.shuffle(data);
        let inputs = data.map(x => x.hp);
        let labels = data.map(x => x.mpg);
        inputs = tf.tensor2d(inputs,[inputs.length,1]);
        inputs_max = inputs.max().dataSync();
        inputs = inputs.div(inputs_max);
        labels = tf.tensor2d(labels,[labels.length,1]);
        labels_max = labels.max().dataSync();
        labels = labels.div(labels_max);
        return {
            inputs : inputs,
            labels : labels,
        }
    })
}

// Model을 생성하는 함수
function createModel(){
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape:[1], units: 16, activation: 'relu'}));
    model.add(tf.layers.dense({units: 52, activation:'relu'}));
    model.add(tf.layers.dense({units: 1,useBias: true}));
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'meanSquaredError',
        metrics: ['mse']
    });
    return model;
}

// Model 훈련
async function trainModel(model, x_train, y_train){
    const batchsize = 32;
    const epochs = 100;
    await model.fit(x_train, y_train,{
        batchsize,
        epochs,
        shuffle: true,
        callbacks:tfvis.show.fitCallbacks(
            {name:'Training 성능'},
            ['mse'],
            {height: 200, callbacks: ['onEpochEnd']}
        )
    })

}

// Model test
async function testModel(model, x_test, y_test){
    let preds = model.predict(x_test).dataSync();
    y_test = y_test.dataSync();
    x_test = x_test.dataSync();
    const series1 = [];
    const series2 = [];
    for(let i = 0 ; i<y_test.length; i++)
    {
        series1.push({x: x_test[i], y:y_test[i]});// 실제
        series2.push({x: x_test[i], y: preds[i]});// 예측
    }
    console.log(series1, series2);
    tfvis.render.scatterplot(
        {name: '산정도'},
        {values:[series1, series2], series:['real data','prediction']},
        {xLabel: '마력', yLabel:'연비',height: 200}
    )
} 

// running function
async function run(){
    const data = await getData();
    const train_size = Math.floor(data.inputs.shape[0] * 0.8);
    const test_size = data.inputs.shape[0] - train_size;
    const[ x_train, x_test] = data.inputs.split([train_size,test_size]);
    const[ y_train, y_test] = data.labels.split([train_size,test_size]);
    //console.log(x_train.shape, x_test.shape);

    const model = createModel();
    // model summary view 
    tfvis.show.modelSummary({name:'Model Summary'},model);

    await trainModel(model, x_train, y_train);

    testModel(model, x_test, y_test);
}

run();