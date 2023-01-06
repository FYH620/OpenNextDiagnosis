<template>
  <v-app id="inspire">
    <v-app-bar
      app
      dark
      elevation="3"
      color="primary"
      icons-and-text
    >
      <v-toolbar-title style="text-overflow:initial;overflow:visible;">Next Diagnosis</v-toolbar-title>

      <v-tabs v-model="tab" centered class="ml-n9">
        <v-tab key="tab-home">
          HOME 首页
          <v-icon>mdi-home</v-icon>
        </v-tab>

        <v-tab key="tab-mask">
          Mask-Detect 实时口罩检测
          <v-icon>mdi-face-mask</v-icon>
        </v-tab>

        <v-tab key="tab-covid">
          Diagnosis-COVID 新冠CT读片诊断
          <v-icon>mdi-doctor</v-icon>
        </v-tab>
      </v-tabs>

      <v-btn
        icon
        @click="changeDarkTheme()"
      >
        <v-icon>mdi-theme-light-dark</v-icon>
      </v-btn>
      <v-spacer></v-spacer>

      <v-avatar
        size="40"
      >
        <img alt="ICON" src="@/assets/logo.png" />
      </v-avatar>
    </v-app-bar>

    <v-tabs-items v-model="tab">
      <v-tab-item key="tab-home" class="bgcolor">
        <Home></Home>
      </v-tab-item>

      <v-tab-item key="tab-mask" class="bgcolor">
        <MaskDetect></MaskDetect>
      </v-tab-item>

      <v-tab-item key="tab-covid" class="bgcolor">
        <DiagnosisCOVID></DiagnosisCOVID>
      </v-tab-item>
    </v-tabs-items>
    
    <v-footer padless class="text-center">
      <v-card
        flat
        tile
        width="100%"
        class="grey darken-2 text-center"
      >
      <v-card-text class="white--text">
        {{ new Date().getFullYear() }} — <strong>NextDiagnosis</strong>
      </v-card-text>
      </v-card>
    </v-footer>
  </v-app>
</template>

<script>
import Home from "@/views/HomeView.vue";
import MaskDetect from "@/views/MaskDetect.vue";
import DiagnosisCOVID from "@/views/DiagnosisCOVID.vue";

export default {
  name: "main-page",

  components: {
    Home,
    MaskDetect,
    DiagnosisCOVID,
  },

  data() {
    return {
      tab: null,
    };
  },

  methods:{
    changeDarkTheme() {
      let isDark = !this.$vuetify.theme.dark;
      this.$vuetify.theme.dark = isDark;
    },
  }
};
</script>