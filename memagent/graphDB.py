import networkx as nx
from typing import List
import difflib


# 3. 模拟图数据库 (基于 NetworkX)
class EnhancedGraphDB:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    # 添加三元组: [[Head, Relation, Tail], ...]
    def add_triples(self, triples: List[List[str]]):
        for item in triples:
            if len(item) == 3:
                head, relation, tail = item
                # 标准化：去除首尾空格，转小写（可选）
                head = head.strip()
                tail = tail.strip()
                self.graph.add_edge(head, tail, relation=relation)
                print(f"      [Graph Write] ({head}) --[{relation}]--> ({tail})")

    # 模糊匹配找到最相似的节点
    def _fuzzy_match_node(self, query_entity: str) -> str:
        all_nodes = list(self.graph.nodes())
        # 使用 difflib 查找最接近的匹配，cutoff=0.6 表示相似度至少 60%
        matches = difflib.get_close_matches(query_entity, all_nodes, n=1, cutoff=0.6)

        # 输出匹配结果
        if matches:
            print(f"      [Graph Match] '{query_entity}' -> 映射为 -> '{matches[0]}'")
            return matches[0]
        return None

    # 支持多跳查询
    def get_context(self, entities: List[str], depth=2) -> str:
        found_nodes = []
        for e in entities:
            matched_node = self._fuzzy_match_node(e)
            if matched_node:
                found_nodes.append(matched_node)
        
        if not found_nodes:
            return ""

        result_lines = set()
        for node in found_nodes:
            # 获取以 node 为中心，半径为 depth 的子图
            # radius=1 找直接邻居，radius=2 找邻居的邻居
            try:
                subgraph = nx.ego_graph(self.graph, node, radius=depth)
                
                # 遍历子图中的边，生成文本
                for u, v, data in subgraph.edges(data=True):
                    rel = data.get('relation', 'related_to')
                    result_lines.add(f"- {u} {rel} {v}")
            except:
                pass
                
        return "\n".join(list(result_lines))