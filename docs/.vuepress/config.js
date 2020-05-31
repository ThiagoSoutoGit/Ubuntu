module.exports = {
    plugins: [
        'vuepress-plugin-latex'
    ],
    base: '/Ubuntu/',
    title: 'Ubuntu Rules',
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