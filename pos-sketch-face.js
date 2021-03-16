let img;
let poseNet;
let pose;

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, () => console.log("model loaded"));
  poseNet.on("pose", (result) => (pose = result[0].pose));
}

function draw() {
  image(video, 0, 0);
  if (pose) {
    console.log(pose);
    fill(255, 255, 255);
    ellipse(pose.rightEye.x, pose.rightEye.y, 32);
    ellipse(pose.leftEye.x, pose.leftEye.y, 32);
    fill(0, 0, 0);
    ellipse(pose.leftEye.x, pose.leftEye.y, 16);
    ellipse(pose.rightEye.x, pose.rightEye.y, 16);
    fill(255, 0, 0);
    ellipse(pose.nose.x, pose.nose.y, 32);
    fill(0, 0, 255);
    rectMode(CENTER);
    rect(pose.leftEar.x, pose.leftEar.y, 16, 32);
    rect(pose.rightEar.x, pose.rightEar.y, 16, 32);
  }
}

