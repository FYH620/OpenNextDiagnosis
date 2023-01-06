<template>
    <div class="text-center">
      <v-dialog :value="show" width="500">
        <v-card :color="bgcolor" dark>
          <v-app-bar flat style="background: transparent !important">
            <div v-if="processing == 'true'">
              <v-progress-circular
                indeterminate
                color="white"
              ></v-progress-circular
              >&nbsp;&nbsp;
            </div>

            <v-app-bar-title class="text-h6 pl-0">
              {{title}}
            </v-app-bar-title>

            <v-spacer></v-spacer>

            <v-btn color="white" icon @click="closeDialog()">
              <v-icon>mdi-close</v-icon>
            </v-btn>
          </v-app-bar>

          <v-card-text>
            <slot />
          </v-card-text>
        </v-card>
      </v-dialog>
    </div>
</template>

<script>
export default {
  name: "StatusDialog",

  props: ['title', 'processing', 'bgcolor', 'show'],

  methods:{
    closeDialog()
    {
      this.$emit('close');
    }
  }
};
</script>
