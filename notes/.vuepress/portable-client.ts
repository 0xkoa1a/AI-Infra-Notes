import { defineClientConfig } from "vuepress/client"

import PortableLayout from "./components/PortableLayout.vue"

export default defineClientConfig({
  layouts: {
    Layout: PortableLayout,
  },
})
