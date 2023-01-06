import Vue from 'vue'
import VueRouter from 'vue-router'
//import Welcome from '../views/WelcomeView.vue'

Vue.use(VueRouter)

const routes = [
  // {
  //   path: '/',
  //   name: 'welcome',
  //   component: Welcome
  // },
  // {
  //   path: '/mask-detect',
  //   name: 'mask-detect',
  //   // layz-loaded
  //   component: () => import(/* webpackChunkName: "about" */ '../views/MaskDetect.vue')
  // },
  // {
  //   path: '/diagnosis-covid',
  //   name: 'diagnosis-covid',
  //   component: () => import(/* webpackChunkName: "about" */ '../views/DiagnosisCOVID.vue')
  // }
]

const router = new VueRouter({
  routes
})

export default router
