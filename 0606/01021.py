from pyecharts import options as opts
from pyecharts.charts import Graph

# 定义节点
nodes = [
    {"name": "AI", "symbolSize": 50},
    {"name": "Machine Learning", "symbolSize": 40},
    {"name": "Deep Learning", "symbolSize": 30},
    {"name": "Natural Language Processing", "symbolSize": 40},
    {"name": "Computer Vision", "symbolSize": 40},
    {"name": "Reinforcement Learning", "symbolSize": 30},
    {"name": "Generative Models", "symbolSize": 30},
]

# 定义边
links = [
    {"source": "AI", "target": "Machine Learning"},
    {"source": "Machine Learning", "target": "Deep Learning"},
    {"source": "Machine Learning", "target": "Natural Language Processing"},
    {"source": "Machine Learning", "target": "Computer Vision"},
    {"source": "Machine Learning", "target": "Reinforcement Learning"},
    {"source": "Deep Learning", "target": "Generative Models"},
]

# 创建图谱
graph = (
    Graph()
    .add("", nodes, links, repulsion=8000)
    .set_global_opts(title_opts=opts.TitleOpts(title="知识图谱示例"))
)

# 渲染图谱到 HTML 文件
graph.render("knowledge_graph.html")
