import Vue from 'vue';
import Vuetify from 'vuetify/lib/framework';
import lightTheme from '@/theme/light';
import darkTheme from '@/theme/dark';

Vue.use(Vuetify);

export default new Vuetify({
    theme: {
        themes: {
            light: lightTheme,
            dark: darkTheme,
        },
    },
});
