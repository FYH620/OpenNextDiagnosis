<template>
  <v-main>
    <v-container>
      <v-row>
        <v-col cols="12" sm="4">
          <v-card elevation="4" rounded="lg" min-height="268">
            <!-- 左侧栏 -->
            <v-img
              height="200px"
              src="@/assets/bg2.png"
            >
              <v-app-bar flat color="rgba(0, 0, 0, 0)">
                <v-btn icon><v-icon color="white">mdi-star-outline</v-icon></v-btn>

                <v-toolbar-title class="text-h6 white--text pl-0">
                  Diagnosis - COVID
                </v-toolbar-title>

                <v-spacer></v-spacer>

                <v-btn color="white" icon @click="helpDialog()">
                  <v-icon>mdi-lightbulb-on-outline</v-icon>
                </v-btn>
              </v-app-bar>

              <v-card-title class="white--text mt-8">
                <v-avatar size="56">
                  <img alt="ICON" src="@/assets/lungs.png" />
                </v-avatar>
                <p class="ml-3">新冠CT读片诊断工具</p>
              </v-card-title>
            </v-img>

            <v-card-text>
              <div class="font-weight-bold ml-8 mb-2">
                <span>💊&nbsp;&nbsp;&nbsp;AI赋能，新冠快速诊断解决方案</span>
              </div>

              <v-timeline align-top dense>
                <v-timeline-item
                  key="1:00am"
                  color="deep-purple lighten-1"
                  small
                >
                  <div>
                    <div class="font-weight-normal">
                      <strong>快速诊断</strong>
                    </div>
                    <div>GPU环境下极速产生结果，为您省时</div>
                  </div>
                </v-timeline-item>

                <v-timeline-item key="2:00am" color="green" small>
                  <div>
                    <div class="font-weight-normal">
                      <strong>高精确度</strong>
                    </div>
                    <div>98.5%准确率，最大程度避免误诊</div>
                  </div>
                </v-timeline-item>

                <v-timeline-item key="3:00am" color="blue" small>
                  <div>
                    <div class="font-weight-normal">
                      <strong>高效模型</strong>
                    </div>
                    <div>ResNet残差网络，高效快速诊断</div>
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
              <v-card-title> 🩺 新冠CT快速诊断平台 </v-card-title>
              <v-card-subtitle> 项目基于ResNet50卷积网络进行肺炎胸片分类，快速精确获得结果 </v-card-subtitle>
              <v-subheader>
                <label for="file"
                  >点击上传按钮上传CT扫描图片，或者拖拽图片文件至框框中<br />将会自动上传并计算结果
                  👇</label
                >
              </v-subheader>
              <input
                id="file"
                type="file"
                ref="uploadInputRef"
                style="display: none"
                :accept="allowFormat"
                @change="handleUpload"
              />
              <div
                class="upload-dragger"
                :class="{
                  'is-dragover': dragover,
                }"
                @drop="handleDropFile"
                @dragenter="handleSuppress"
                @dragover="handleSuppress"
                @dragleave.prevent="dragover = false"
                @click="$refs.uploadInputRef.click()"
              >
                拖拽CT图片至此 或者点击 <em>上传</em>
              </div>
              <div class="upload-tip">
                上传文件大小不超过 {{ fileSizeLimit }} MB
              </div>
            </div>
          </v-card>
        </v-col>
      </v-row>
    </v-container>

    <!-- Dialog -->
    <StatusDialog
      title="检测运行中..."
      processing="true"
      bgcolor="orange darken-2"
      :show="this.dialogStatus == 'Running'"
      v-on:close="cancelTask()"
    >
      检测正在运行中，可能花费时间较长，请稍安勿躁...
    </StatusDialog>

    <StatusDialog
      title="⚠️⚠️ 警告 检测到阳性"
      bgcolor="red"
      :show="this.dialogStatus == 'Positive'"
      v-on:close="cancelTask()"
    >
      检测到阳性，请立即处理！
    </StatusDialog>

    <StatusDialog
      title="✅ 结果为阴性"
      bgcolor="green"
      :show="this.dialogStatus == 'Negative'"
      v-on:close="cancelTask()"
    >
      检测结果为阴性。10秒后窗口将自动关闭
    </StatusDialog>

    <StatusDialog
      title="❌ 发生错误"
      bgcolor="orange darken-2"
      :show="this.dialogStatus == 'Error'"
      v-on:close="cancelTask()"
    >
      检测过程中发生错误。错误码：{{errorText}}
    </StatusDialog>

    <StatusDialog
      title="🔎 了解 Diagnosis COVID..."
      bgcolor="blue lighten-1"
      :show="this.dialogStatus == 'Help'"
      v-on:close="cancelTask()"
    >
      <div class="font-weight-bold">
        <li>通过AI技术加速确诊病例的判断</li>
        <li>与抗原检测试剂等技术交叉验证，缩短确诊时间</li>
        <li>无接触式快速排查，减少患者集中管控的交叉感染风险</li>
      </div>
    </StatusDialog>
  </v-main>
</template>


<style lang="scss" scoped>
.upload {
  &-dragger {
    width: 400px;
    height: 180px;
    border: 2px dashed rgba(156, 146, 146, 0.63);
    border-radius: 6px;
    box-sizing: border-box;
    text-align: center;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    line-height: 180px;
    margin: 20px auto 10px;
    font-size: 20px;
    &:hover {
      box-shadow: 0px 0px 10px #4587dd;
    }
  }
  em {
    color: #409eff;
    font-style: normal;
  }
  &-tip {
    margin-top: 10px;
    font-size: 13px;
    color: #606266;
    text-align: center;
  }
  .is-dragover {
    background-color: rgba(238, 240, 242, 0.5);
    border: 2px dashed #83cfc9;
  }
}
</style>

<script>
import StatusDialog from '@/components/StatusDialog.vue'

export default {
  name: "DiagnosisCOVID",

  components: {
    StatusDialog
  },

  data() {
    return {
      fileSizeLimit: this.$config.fileUpload.maxSizeLimit,
      allowFormat: this.$config.fileUpload.allowFormat.map((x) => `.${x}`).join(","),
      errorText: "",
      dialogTimeoutClose: 0,
      
      //dialog: "", "Running", "Positive", "Negative", "Error", "Help"
      dialogStatus: "",

      //status
      loading: false,
      dragover: false,
    };
  },

  methods: {
    checkFileSize(file) {
      const isLimit = file.size / 1024 / 1024 <= this.fileSizeLimit;
      if (!isLimit) {
        this.errorResult("-1", `文件上传失败：上传图片大小不能超过 ${this.fileSizeLimit} MB`);
      }
      return isLimit;
    },
    handleSuppress(e) {
      e.stopPropagation();
      e.preventDefault();
      this.dragover = true;
    },
    handleUpload(e) {
      this.loading = true;
      const files = e.target.files || [];
      if (files && files[0] && this.checkFileSize(files[0])) {
        this.upload(files[0]);
      }
    },
    handleDropFile(e) {
      e.stopPropagation();
      e.preventDefault();
      this.dragover = false;
      const files = e.dataTransfer.files || [];
      if (files && files[0]) {
        this.upload(files[0]);
      }
    },
    upload(file) {
      let forms = new FormData();
      let configs = {
        headers: { "Content-Type": "multipart/form-data" },
      };
      forms.append("file", file); // 获取上传图片信息
      this.$axios
        .post("/api/diagnosis-covid/upload", forms, configs)
        .then((res) => {
          //console.log(res);
          if (res.data.code != 0) this.errorResult(res.data.code, res.data.msg);
          else if (res.data.positive) this.positiveResult();
          else this.negativeResult();
          this.loading = false;
        });
      this.showDialog();
    },
    showDialog() {
      this.dialogStatus = "Running";
    },
    positiveResult() {
      this.dialogStatus = "Positive";
    },
    negativeResult() {
      this.dialogStatus = "Negative";
      this.dialogTimeoutClose = setTimeout(() => {
        this.dialogTimeoutClose = null;
        if(this.dialogStatus == "Negative")
        {
          this.dialogStatus = "";
          this.loading = false;
        }
      }, 10000); //10s
    },
    errorResult(errorCode, msg) {
      this.errorText = String(errorCode) + "\n" + msg;
      this.dialogStatus = "Error";
    },
    cancelTask() {
      if (this.dialogTimeoutClose) {
        clearTimeout(this.dialogTimeoutClose);
        this.dialogTimeoutClose = null;
      }
      this.dialogStatus = "";
      this.loading = false;
    },
    helpDialog() {
      this.dialogStatus = "Help";
    },
  },
};
</script>