import Vue from 'vue'
import App from './App.vue'
import router from './router'
import vuetify from './plugins/vuetify'
import axios from 'axios'
import globalConfig from './config/config'

Vue.config.productionTip = false

//axios
axios.defaults.withCredentials = true;
axios.defaults.crossDomain = true;
Vue.prototype.$axios = axios;

//config
Vue.prototype.$config = globalConfig;

new Vue({
  router,
  vuetify,
  render: h => h(App)
}).$mount('#app')
