import DefaultTheme from 'vitepress/theme'
import './style/index.css'
import ArticleMetadata from "./components/ArticleMetadata.vue"

export default {
    extends: DefaultTheme,
    enhanceApp({ app }) {
        app.component('ArticleMetadata', ArticleMetadata)
    }
}