import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/CourseGraph/',
  title: "CourseGraph",
  titleTemplate: ':title',
  lastUpdated: true,
  description: "使用大模型自动构建课程知识图谱",
  head: [['link', { rel: 'icon', href: '/CourseGraph/logo.png' }]],
  themeConfig: {
    search: {
      provider: 'local',
    },
    nav: [
      { text: '主页', link: '/' },
      {
        text: '指南', items: [
          { text: '环境搭建', link: '/tutorials/env' },
          { text: '知识图谱抽取', link: '/tutorials/kg' },
          { text: '为实体设置资源', link: '/tutorials/resource' },
          { text: '文档解析器', link: '/tutorials/parser' },
          { text: '大模型', link: '/tutorials/llm' },
          { text: 'example管理', link: '/tutorials/example' },
          { text: '数据库', link: '/tutorials/kb' },
          { text: 'Agent', link: '/tutorials/agent/agent' }]
      },
      { text: 'API 参考', link: '/api-reference' }
    ],
    logo: '/logo.png',
    footer: {
      message: 'Released under the  Apache 2.0 License.',
      copyright: `Copyright © 2024-${new Date().getFullYear()} present Wang Tao`,
    },

    sidebar: [
      {
        text: '前言', items: [
          { text: '介绍', link: '/introduce' }
        ], collapsed: false
      },
      {
        text: '指南', items: [
          { text: '环境搭建', link: '/tutorials/env' },
          { text: '知识图谱抽取', link: '/tutorials/kg' },
          { text: '为实体设置资源', link: '/tutorials/resource' },
          { text: '文档解析器', link: '/tutorials/parser' },
          { text: '大模型', link: '/tutorials/llm' },
          { text: 'example管理', link: '/tutorials/example' },
          { text: '数据库', link: '/tutorials/kb' },
          {
            text: 'Agent', items: [
              { text: 'Agent编排框架', link: '/tutorials/agent/agent' },
              { text: '使用Agent抽取知识图谱', link: '/tutorials/agent/kg' }
            ]
          },
          {
            text: '其他', items: [
              { text: 'Rust扩展', link: '/tutorials/other/rust' },
            ]
          }
        ], collapsed: false
      },
      { text: 'API 参考', link: '/api-reference' }
    ],

    editLink: {
      pattern: 'https://github.com/wangtao2001/CourseGraph/edit/dev/docs/:path',
      text: '在GitHub编辑本页'
    },

    outline: {
      level: [2, 4],
      label: 'On this page'
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/wangtao2001/CourseGraph' },

      // {
      //   icon: {
      //     svg: '<?xml version="1.0" encoding="UTF-8"?><svg width="24" height="24" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 39H44V24V9H24H4V24V39Z" fill="none" stroke="#333" stroke-width="4" stroke-linejoin="round"/><path d="M4 9L24 24L44 9" stroke="#333" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><path d="M24 9H4V24" stroke="#333" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><path d="M44 24V9H24" stroke="#333" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
      //   },
      //   link: 'mailto:wangtao.cpu@gmail.com',
      //   ariaLabel: 'mail'
      // }

    ]
  }
})
