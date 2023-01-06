<template>
  <v-main>
    <v-container>
      <v-row>
        <v-col cols="12" sm="4">
          <v-card elevation="4" rounded="lg" min-height="268">
            <!-- 左侧栏 -->
            <v-img height="200px" src="@/assets/bg1.png">
              <v-app-bar flat color="rgba(0, 0, 0, 0)">
                <v-btn icon><v-icon color="white">mdi-star-outline</v-icon></v-btn>

                <v-toolbar-title class="text-h6 white--text pl-0">
                  MaskDetect
                </v-toolbar-title>

                <v-spacer></v-spacer>

                <v-btn color="white" icon @click="helpDialog()">
                  <v-icon>mdi-lightbulb-on-outline</v-icon>
                </v-btn>
              </v-app-bar>

              <v-card-title class="white--text mt-8">
                <v-avatar size="56">
                  <img alt="ICON" src="@/assets/facemask.png" />
                </v-avatar>
                <p class="ml-3">实时画面口罩检测工具</p>
              </v-card-title>
            </v-img>

            <v-card-text>
              <div class="font-weight-bold ml-8 mb-2">
                <span
                  >🎥&nbsp;&nbsp;&nbsp;实时监测，为所有人的安全保驾护航</span
                >
              </div>

              <v-timeline align-top dense>
                <v-timeline-item
                  key="1:00am"
                  color="deep-purple lighten-1"
                  small
                >
                  <div>
                    <div class="font-weight-normal">
                      <strong>实时检测</strong>
                    </div>
                    <div>连续拍摄，实时告警<br />第一时间发现漏网之鱼</div>
                  </div>
                </v-timeline-item>

                <v-timeline-item key="2:00am" color="green" small>
                  <div>
                    <div class="font-weight-normal">
                      <strong>高效算法</strong>
                    </div>
                    <div>快速高精度的 SSD<br />目标检测模型</div>
                  </div>
                </v-timeline-item>

                <v-timeline-item key="3:00am" color="blue" small>
                  <div>
                    <div class="font-weight-normal">
                      <strong>鲁棒性强</strong>
                    </div>
                    <div>
                      准确率与召回率达到高度平衡<br />在各种干扰环境下均可以完美工作
                    </div>
                  </div>
                </v-timeline-item>
              </v-timeline>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" sm="8">
          <v-card elevation="4" min-height="70vh" rounded="lg">
            <!-- 主要栏目 -->
            <div class="upload">
              <v-card-title> 📹 口罩实时监测平台 </v-card-title>
              <v-card-subtitle>
                项目基于SSD目标检测网络进行多重优化，适应黑暗与雾天等多种干扰环境
              </v-card-subtitle>

              <v-subheader>
                <label
                  >点击开始按钮，开始录像并实时检测，如果发现未戴口罩的人会自动记录并告警<br />点击停止按钮停止录像👇</label
                >
              </v-subheader>

              <v-container>
                <v-row>
                  <v-col cols="12" sm="9">
                    <canvas
                      ref="hiddenCanvasRef"
                      style="display: none"
                    ></canvas>
                    <video
                      ref="videoRef"
                      id="videoEle"
                      width="100%"
                      height="100%"
                      muted
                      controls
                      style="display: none"
                    ></video>
                    <canvas
                      ref="videoCanvasRef"
                      id="videoCanvas"
                      width="100%"
                      height="100%"
                    ></canvas>
                  </v-col>
                  <v-col cols="12" sm="3">
                    <v-container>
                      <v-row>
                        <v-col>
                          <v-btn
                            :loading="loadingPlay"
                            :disabled="!startEnabled"
                            color="green"
                            class="ma-2 white--text"
                            @click="startRecord()"
                          >
                            开始录像
                            <v-icon right dark> mdi-play-circle </v-icon>
                          </v-btn>

                          <v-btn
                            :loading="loadingStop"
                            :disabled="!endEnabled"
                            color="red"
                            class="ma-2 white--text"
                            @click="endRecord()"
                          >
                            停止录像
                            <v-icon right dark> mdi-stop </v-icon>
                          </v-btn>

                          <v-switch
                            v-model="foggyMode"
                            label="雾天模式"
                            color="red darken-3"
                            value="off"
                            true-value="on"
                            false-value="off"
                            hide-details
                            @change="foggyChanged()"
                          ></v-switch>

                          <v-switch
                            v-model="darkMode"
                            label="黑暗模式"
                            color="indigo darken-3"
                            true-value="on"
                            false-value="off"
                            value="off"
                            hide-details
                            @change="darkChanged()"
                          ></v-switch>

                          <v-switch
                            v-model="warningSound"
                            label="启用报警声音"
                            color="light-blue darken-4"
                            true-value="on"
                            false-value="off"
                            value="on"
                            hide-details
                          ></v-switch>

                        </v-col>
                      </v-row>
                      <v-row>
                        <v-col>
                          
                          <v-divider></v-divider>
                          <v-switch
                            v-model="mirrorMode"
                            label="画面镜像"
                            color="teal darken-2"
                            true-value="on"
                            false-value="off"
                            value="off"
                            hide-details
                            @change="mirrorChanged()"
                          ></v-switch>

                        </v-col>
                      </v-row>
                    </v-container>
                  </v-col>
                </v-row>

                <v-row>
                  <v-col>
                    <v-select
                      ref="cameraSelector"
                      prepend-icon="mdi-video"
                      v-model="cameraSelected"
                      item-text="label"
                      item-value="deviceId"
                      :items="cameras"
                      return-object
                      label="选择工作摄像头..."
                      hint="当前工作摄像头"
                      persistent-hint
                      solo
                      @change="cameraChanged()"
                    ></v-select>
                  </v-col>
                </v-row>
              </v-container>
            </div>
          </v-card>
        </v-col>
      </v-row>
    </v-container>

    <StatusDialog
      title="🔎 了解 MaskDetect..."
      bgcolor="blue lighten-1"
      :show="this.dialogStatus == 'Help'"
      v-on:close="closeDialog()"
    >
      <div class="font-weight-bold">
        <li>在视频流中监控所有人员的口罩佩戴情况，并实时报警</li>
        <li>有效阻止病毒空气传播风险，方便政府/企业进行人流管控和返工复产</li>
        <li>未来将接入基于图像分析的无接触体温检测系统，医疗机器人外呼高效筛查等系统，便于疫区的管控和非疫区的防控</li>
      </div>
    </StatusDialog>

    <StatusDialog
      title="❌ 发生错误"
      bgcolor="orange darken-2"
      :show="this.dialogStatus == 'Error'"
      v-on:close="closeDialog()"
    >
      检测过程中发生错误。错误码：{{errorText}}
    </StatusDialog>

    <audio ref="warningAudio" loop>
      <source src="@/assets/warning.wav" type="audio/wav">
      您的浏览器不支持音频功能！
    </audio>
  </v-main>
</template>

<script>
import StatusDialog from '@/components/StatusDialog.vue'

export default {
  name: "MaskDetect",

  components: {
    StatusDialog
  },

  data() {
    return {
      //status
      isWorking: false,
      startEnabled: true,
      endEnabled: false,
      loadingPlay: false,
      loadingStop: false,

      //video
      mediaConstraints: {
        audio: false,
        video: {}
      },
      workingStream: {},
      squares: [],

      //camera select
      cameraSelected: {},
      cameras: [],

      //dialog: "", "Running", "Positive", "Negative", "Error", "Help"
      dialogStatus: "",
      errorText: "",

      //warningSound
      isPlayingWarning: false,

      //options
      foggyMode: "off",
      darkMode: "off",
      mirrorMode: "off",
      warningSound: "on",

      //elements
      hiddenCanvas: {},
      video: {},
      videoCanvas: {},
      ctx: {},
      realWidth: 0,
      realHeight: 0,
      drawPlayId: 0,
      maskCheckLoopId: 0,
    };
  },

  methods: {
    foggyChanged(){
      if(this.foggyMode == "on")
      {
        this.darkMode = "off";
        this.$config.maskDetectInterval = this.$config.maskDetectIntervalLarge;
      }
      else
        this.$config.maskDetectInterval = this.$config.maskDetectIntervalNormal;
    },
    darkChanged(){
      if(this.darkMode == "on")
        this.foggyMode = "off";
    },
    mirrorChanged(){
      ;
    },
    enumerateDevices(){
      navigator.mediaDevices.enumerateDevices()
        .then((devices) => {
          devices.forEach((device) => {
            if(device.kind == 'videoinput'){
              this.cameras.push({
                  'label': device.label,
                  'deviceId': device.deviceId
              });
            }
          });
          if(devices.length > 0)
          {
            this.cameraSelected = this.cameras[0];
            this.cameraChanged();
          }
          if(devices.length == 1)
            this.$refs.cameraSelector.disabled = true;
        })
        .catch((err) => {
          console.log(err.name + ": " + err.message);
        });
    },
    cameraChanged(){
      this.mediaConstraints.video.deviceId = { exact: this.cameraSelected.deviceId };
      if(this.isWorking)
      {
        this.endRecord();
        this.startRecord();
      }
    },
    startRecord() {
      if (this.isWorking) return;
      this.startEnabled = false;
      this.loadingPlay = true;

      navigator.mediaDevices
        .getUserMedia(this.mediaConstraints)
        .then(this.onMediaSuccess)
        .catch(this.onMediaError);
    },
    endRecord() {
      this.endMaskCheckLoop();
      this.video.pause();

      this.isWorking = false;
      this.endEnabled = false;
      this.startEnabled = true;
      if (this.workingStream) {
        this.workingStream.getVideoTracks().forEach(track => {
          track.stop();
        });
        this.workingStream = null;
      }
    },
    onMediaSuccess(stream) {
      this.workingStream = stream;

      this.video.controls = false;
      this.video.srcObject = stream;
      this.video.play();

      this.isWorking = true;
      this.endEnabled = true;
      this.loadingPlay = false;

      //this.resizeCanvas(this.video.videoWidth, this.video.videoHeight);
      this.playCanvas();
      this.startMaskCheckLoop();
    },
    onMediaError(err) {
      this.endRecord();
      this.errorDialog(-1, "摄像头开启失败：" + str(err));
    },
    resizeCanvas(width, height) {
      if (!width) {
        let parentStyle = window.getComputedStyle(this.videoCanvas.parentNode);
        width = parseInt(parentStyle.width);
      }
      //if(!height)
      height = (width * 6) / 8;

      this.videoCanvas.width = width;
      this.videoCanvas.height = height;
      this.hiddenCanvas.width = width;
      this.hiddenCanvas.height = height;

      this.realWidth = this.videoCanvas.width;
      this.realHeight = this.videoCanvas.height;
    },
    playCanvas() {
      //video
      if(this.mirrorMode == "on")
      {
        this.ctx.save();
        this.ctx.translate(this.realWidth, 0);
        this.ctx.scale(-1, 1);
      }
      this.ctx.drawImage(this.video, 0, 0, this.realWidth, this.realHeight);
      if(this.mirrorMode == "on")
      {
        this.ctx.restore();
      }

      //squares & confidence
      this.squares.forEach((s) => {
        if (!s) return;

        this.ctx.lineWidth = "2"; //框的粗细
        if (s.type == "mask")
          this.ctx.strokeStyle = this.ctx.fillStyle = "green";
        else this.ctx.strokeStyle = this.ctx.fillStyle = "red";

        let text = s.type == "mask" ? "Mask " : "NoMask ";
        if (s.confidence) text += s.confidence.toString();

        this.ctx.font = "bold 13px Arial";
        this.ctx.fillText(text, s.ax, s.ay - 5);
        this.ctx.strokeRect(s.ax, s.ay, s.bx - s.ax, s.by - s.ay);
      });

      //loop
      this.drawPlayId = window.requestAnimationFrame(() => {
        if (!this.isWorking) {
          cancelAnimationFrame(this.drawPlayId);
          this.drawPlayId = 0;
        } else this.playCanvas();
      });
    },
    clearCanvas() {
      this.ctx.fillStyle = "#000000";
      this.ctx.fillRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
    },
    playWarningSound(){
      if(this.warningSound == "on" && this.isWorking && !this.isPlayingWarning)
      {
        this.$refs.warningAudio.play();
        this.isPlayingWarning = true;
      }
    },
    stopWarningSound(){
      if(this.isPlayingWarning)
      {
        this.$refs.warningAudio.pause();
        this.isPlayingWarning = false;
      }
    },
    startMaskCheckLoop() {
      this.maskCheckLoopId = setTimeout(() => {
        if(!this.isWorking)
        {
          this.endMaskCheckLoop();
          return;
        }
        this.screenshotAndSend();
        this.startMaskCheckLoop();
      }, this.$config.maskDetectInterval);
    },
    endMaskCheckLoop() {
      clearInterval(this.maskCheckLoopId);
      this.maskCheckLoopId = 0;
    },
    //axios
    screenshotAndSend() {
      const ctx = this.hiddenCanvas.getContext("2d");
      const width = this.realWidth;
      const height = this.realHeight;

      if(this.mirrorMode == "on")
      {
        ctx.save();
        ctx.translate(width, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(this.video, 0, 0, width, height);
      if(this.mirrorMode == "on")
      {
        ctx.restore();
      }

      this.upload(this.hiddenCanvas.toDataURL("image/png"), width, height);
    },
    upload(image, width, height) {
      let configs = {
        headers: { "Content-Type": "application/json" },
      };
      let data = {
        image: image.split(",")[1],
        width: width,
        height: height,
        foggyMode: this.foggyMode == "on" ? 1 : 0,
        darkMode: this.darkMode == "on" ? 1 : 0,
      };

      this.$axios.post("/api/mask-detect/upload", data, configs).then(this.processResult);
    },
    processResult(res){
      //console.log(res);
      if (res.data.code != 0) {
        this.endRecord();
        console.error("Backend Error", res.data.msg);
      }
      else {
        this.squares = res.data.peoples;

        let hasNoMask = false;
        if(this.warningSound == "on" && this.isWorking)
        {
          this.squares.forEach((s) => {
            if (s.type == "nomask")
            {
              hasNoMask = true;
              this.playWarningSound();
            }
          });
        }
        if(!hasNoMask)
          this.stopWarningSound();
      }
    },
    helpDialog() {
      this.dialogStatus = "Help";
    },
    errorDialog(errorCode, msg) {
      this.errorText = String(errorCode) + "\n" + msg;
      this.dialogStatus = "Error";
    },
    closeDialog() {
      this.dialogStatus = "";
    },
  },
  //init
  mounted: function () {
    this.video = this.$refs.videoRef;
    this.hiddenCanvas = this.$refs.hiddenCanvasRef;
    this.videoCanvas = this.$refs.videoCanvasRef;
    this.ctx = this.videoCanvas.getContext("2d");

    this.enumerateDevices();

    this.resizeCanvas();
    this.clearCanvas();
  },
};
</script>