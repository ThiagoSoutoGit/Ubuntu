module.exports = {
    base: '/Ubuntu/',
    title: 'Ubuntu Rules',
    plugins: ['@vuepress/back-to-top'],
    themeConfig: {
        nav: [
            { text: 'TensorFlow', link: '../../Pages/Python/TensorFlow.html' },
            { text: 'Object Detection and Counting', link: '../../Pages/Python/Object-Counting.html' },
        ],
        sidebar: 'auto'
    },
    markdown: {
        lineNumbers: true
    }
}