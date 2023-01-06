<template>
  <div>
    <section id="welcome">
      <v-row no-gutters>
        <v-img
          :max-height="'calc(140vh - ' + $vuetify.application.top + 'px)'"
          src="@/assets/homebg1.jpg"
        >
          <v-theme-provider>
            <v-container fill-height>
              <v-row
                align="center"
                class="mx-auto"
                justify="center"
              >
                <v-col
                  class="text-center"
                  cols="12"
                  tag="h1"
                >
                  <span
                    :class="[$vuetify.breakpoint.smAndDown ? 'display-3': 'display-4']"
                    class="font-weight-black"
                  >
                    Next Diagnosis
                  </span>

                  <br> <br>

                  <span
                  :class="[$vuetify.breakpoint.smAndDown ? 'display-2' : 'display-3']"
                    class="font-weight-light"
                  >
                    疫情智能防控聚合工具包
                  </span> 

                  <br>

                </v-col>

                <v-btn
                  class="align-self-end"
                  fab
                  outlined
                  @click="$vuetify.goTo('#about-me')"
                >
                  <v-icon>mdi-chevron-double-down</v-icon>
                </v-btn>
              </v-row>
            </v-container>
          </v-theme-provider>
        </v-img>
      </v-row>
    </section>

    <section id="about-me">
      <div class="py-12"></div>

      <v-container class="text-center">
        <h2 class="display-2 font-weight-bold mb-3">工具于手，救人于心...</h2>

        <v-responsive
          class="mx-auto mb-8"
          width="56"
        >
          <v-divider class="mb-1"></v-divider>

          <v-divider></v-divider>
        </v-responsive>

        <v-responsive
          class="mx-auto title font-weight-light mb-8"
          max-width="720"
        >
          疫情当下，无人可以独善其身。<br>
          唯有勇敢面对，主动出击，方能寻得破局之策。<br><br>
          NextDignosis，从源头助力急速诊断，为公共安全铸就坚实之墙
        </v-responsive>

        <v-container
          class="text-center"
        >
          <v-row>
            <v-col></v-col>
            <v-col></v-col>

            <v-col>
              <v-avatar
                class="elevation-12 mb-12"
                size="128"
              >
                <v-img src="@/assets/fyh.jpg"></v-img>
              </v-avatar>
            </v-col>

            <v-col>
              <v-avatar
                class="elevation-12 mb-12"
                size="128"
              >
                <v-img src="@/assets/yq.jpg"></v-img>
              </v-avatar>
            </v-col>

            <v-col>
              <v-avatar
                class="elevation-12 mb-12"
                size="128"
              >
                <v-img src="@/assets/wyb.jpg"></v-img>
              </v-avatar>
            </v-col>

            <v-col></v-col>
            <v-col></v-col>
          </v-row>
        </v-container>

        <div></div>

        <v-btn
          color="grey"
          href="#"
          outlined
          large
        >
          <span class="grey--text text--darken-1 font-weight-bold">
            点击查看更多文档帮助
          </span>
        </v-btn>
      </v-container>

      <div class="py-12"></div>
    </section>

    <section id="stats">
      <v-parallax
        :height="$vuetify.breakpoint.smAndDown ? 700 : 500"
        src="@/assets/homebg2.jpg"
      >
        <v-container fill-height>
          <v-row class="mx-auto">
            <v-col
              v-for="[value, title] of stats"
              :key="title"
              cols="12"
              md="4"
            >
              <div class="text-center">
                <div
                  class="display-3 font-weight-black mb-4"
                  v-text="value"
                ></div>

                <div
                  class="title font-weight-regular text-uppercase"
                  v-text="title"
                ></div>
              </div>
            </v-col>
          </v-row>
        </v-container>
      </v-parallax>
    </section>

    <section
      id="features"
      class="grey lighten-3"
    >
      <div class="py-12"></div>

      <v-container class="text-center">
        <h2 class="display-2 font-weight-bold mb-3">特性与优势</h2>

        <v-responsive
          class="mx-auto mb-12"
          width="56"
        >
          <v-divider class="mb-1"></v-divider>

          <v-divider></v-divider>
        </v-responsive>

        <v-row>
          <v-col
            v-for="({ icon, title, text }, i) in features"
            :key="i"
            cols="12"
            md="4"
          >
            <v-card
              class="py-12 px-4"
              color="grey lighten-5"
              flat
            >
              <v-theme-provider dark>
                <div>
                  <v-avatar
                    color="primary"
                    size="88"
                  >
                    <v-icon
                      large
                      v-text="icon"
                    ></v-icon>
                  </v-avatar>
                </div>
              </v-theme-provider>

              <v-card-title
                class="justify-center font-weight-black text-uppercase"
                v-text="title"
              ></v-card-title>

              <v-card-text
                class="subtitle-1"
                v-text="text"
              >
              </v-card-text>
            </v-card>
          </v-col>
        </v-row>
      </v-container>

      <div class="py-12"></div>
    </section>
  </div>
</template>

<script>

export default {
  name: "HomeView",

  data () {
      return {
        features: [
          {
            icon: 'mdi-account-group-outline',
            title: '模型优异',
            text: '对密集目标的检测能力强，有效抵抗雾天、黑暗等干扰环境；模型本身性能优异，计算速度快，肺炎CT诊断判别准确率达98.5%',
          },
          {
            icon: 'mdi-update',
            title: '快速上手',
            text: '界面可视化程度高，接入轻量快速；产品实时性高，部署容易，',
          },
          {
            icon: 'mdi-shield-outline',
            title: '随处可用',
            text: '小区门口、楼道门口，上班场所，交通闸机，娱乐商超，街道闹市......',
          },
        ],
        stats: [
          ['98.5%', '高精度高速度'],
          ['5 min', '急速部署上线'],
          ['1+1>2', '应用互联互通'],
        ],
      }
    }
}
</script>