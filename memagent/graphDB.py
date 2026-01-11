import networkx as nx
from typing import List
import difflib


# 模拟图数据库
class EnhancedGraphDB:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    # 添加三元组: [[Head, Relation, Tail], ...]
    def add_triples(self, triples: List[List[str]]):
        for item in triples:
            if len(item) == 3:
                head, relation, tail = item
                # 标准化：去除首尾空格
                head = head.strip()
                tail = tail.strip()
                relation = relation.strip()
                
                # 查重与合并逻辑
                if self.graph.has_edge(head, tail):
                    # 获取现有的关系数据
                    existing_data = self.graph.get_edge_data(head, tail)
                    existing_relation = existing_data.get('relation', '')
                    
                    # 1如果关系完全一样，直接跳过
                    if relation == existing_relation:
                        continue
                    
                    # 如果关系不一样，进行追加合并
                    if relation not in existing_relation:
                        new_relation = f"{existing_relation}, {relation}"
                        self.graph[head][tail]['relation'] = new_relation
                        print(f"      [Graph Update] ({head}) --[{new_relation}]--> ({tail})")
                else:
                    # 如果是新边，直接添加
                    self.graph.add_edge(head, tail, relation=relation)
                    print(f"      [Graph Write] ({head}) --[{relation}]--> ({tail})")

    # 模糊匹配找到最相似的节点
    def _fuzzy_match_node(self, query_entity: str) -> str:
        all_nodes = list(self.graph.nodes())
        if not all_nodes:
            return None
        # 使用 difflib 查找最接近的匹配，cutoff=0.6 表示相似度至少 60%
        matches = difflib.get_close_matches(query_entity, all_nodes, n=1, cutoff=0.6)

        # 返回匹配结果
        if matches:
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

        # 使用集合避免重复的边描述
        result_lines = set()
        
        # 将图转为无向图，这样入边和出边距离都是1
        undirected_G = self.graph.to_undirected()

        for node in found_nodes:
            try:
                # 找出距离中心节点 depth 范围内的所有节点
                nearby_nodes = nx.single_source_shortest_path_length(undirected_G, node, cutoff=depth).keys()
                
                # 在原始有向图中提取这些节点构成的子图
                subgraph = self.graph.subgraph(nearby_nodes)
                
                # 遍历子图的边，生成文本
                for u, v, data in subgraph.edges(data=True):
                    rel = data.get('relation', 'related_to')
                    result_lines.add(f"- {u} {rel} {v}")
            except Exception as e:
                print(f"[Graph Error] Context retrieval failed: {e}")
                pass
                
        return "\n".join(list(result_lines))
    
    # 清空图数据库，测试时使用
    def clear(self):
        self.graph.clear()
        print("      [Graph Clear] 已清空 NetworkX 图数据")