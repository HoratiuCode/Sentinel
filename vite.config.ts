import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Split the heavy TensorFlow + model libs into their own lazily-loaded
        // chunk so the app shell paints before the 1MB+ ML bundle is fetched.
        manualChunks(id) {
          if (id.includes('@tensorflow')) return 'tensorflow';
          if (id.includes('@tensorflow-models')) return 'tf-models';
        },
      },
    },
  },
})
