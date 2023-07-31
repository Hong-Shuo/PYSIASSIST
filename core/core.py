
from core.commonly_API_names import Pytorch_Data_preprocessing_API,SKlearn_Data_preprocessing_API,Pytorch_activation_functions_API,Pytorch_all_loss_functions_API,Pytorch_loss_functions_API_have_activation_fuctions

commonly_API_names_dict={1:Pytorch_Data_preprocessing_API,2:SKlearn_Data_preprocessing_API,3:Pytorch_activation_functions_API,4:Pytorch_all_loss_functions_API,5:Pytorch_loss_functions_API_have_activation_fuctions}

from scalpel.SSA.const import SSA
import os
import ast
import astor

import astunparse
from scalpel.cfg import CFGBuilder
import graphviz as gv

import astroid

from pylint.checkers import BaseChecker
from pylint.lint import Run
import re
import json

path =None
all_block_dict = None
flattened_cfg =None
def_dict,class_dict,def_cfg_dict,code,all_code_block_index_dict = None,None,None,None,None

isfind =True
isadd =False

class AddPass(ast.NodeTransformer):
    """If the function body is empty, add a pass statement."""
    def visit_FunctionDef(self, node):
        if not node.body:
            node.body = [ast.Pass()]
        return node
    def visit_With(self, node):
        if not node.body:
            node.body = [ast.Pass()]
        return node



def to_src_without_children(node):
    if isinstance(node, ast.FunctionDef):
        node = ast.copy_location(
            ast.FunctionDef(name=node.name, args=node.args, body=[], decorator_list=node.decorator_list), node)
    elif isinstance(node, ast.ClassDef):
        node = ast.copy_location(ast.ClassDef(name=node.name, bases=node.bases, keywords=node.keywords, body=[],
                                              decorator_list=node.decorator_list), node)
    elif isinstance(node, ast.If):
        node = ast.copy_location(ast.If(test=node.test, body=[], orelse=[]), node)
    elif isinstance(node, ast.While):
        node = ast.copy_location(ast.While(test=node.test, body=[], orelse=[]), node)
    elif isinstance(node, ast.For):
        node = ast.copy_location(ast.For(target=node.target, iter=node.iter, body=[], orelse=[]), node)
    elif isinstance(node, ast.Try):
        node = ast.copy_location(ast.Try(body=[], handlers=[], orelse=[], finalbody=[]), node)
    elif isinstance(node, ast.With):
        node = ast.copy_location(ast.With(items=node.items, body=[]), node)

    # Append the line number to the source code
    src_code = astor.to_source(node).strip().replace("\n","")
    src_code += f":Line {node.lineno}\n"
    return src_code


def ast_to_astroid(ast_node):
    ast_node = AddPass().visit(ast_node)
    source_code = astunparse.unparse(ast_node)
    if source_code.startswith("\n"):
        source_code = source_code.lstrip("\n")
    try:
        astroid_node = astroid.parse(source_code)
    except astroid.exceptions.AstroidSyntaxError:
        source_code += "\n  pass"
        astroid_node = astroid.parse(source_code)
    return astroid_node


def to_src_without_children_astroid(node):
    astr_code = node.as_string()
    ast_node = ast.parse(astr_code)
    ast_node = ast_node.body[0]
    if isinstance(ast_node, ast.FunctionDef):
        ast_node = ast.copy_location(
            ast.FunctionDef(name=ast_node.name, args=ast_node.args, body=[], decorator_list=ast_node.decorator_list), ast_node)
    elif isinstance(ast_node, ast.ClassDef):
        ast_node = ast.copy_location(ast.ClassDef(name=ast_node.name, bases=ast_node.bases, keywords=ast_node.keywords, body=[],
                                              decorator_list=ast_node.decorator_list), ast_node)
    elif isinstance(ast_node, ast.If):
        ast_node = ast.copy_location(ast.If(test=ast_node.test, body=[], orelse=[]), ast_node)
    elif isinstance(ast_node, ast.While):
        ast_node = ast.copy_location(ast.While(test=ast_node.test, body=[], orelse=[]), ast_node)
    elif isinstance(ast_node, ast.For):
        ast_node = ast.copy_location(ast.For(target=ast_node.target, iter=ast_node.iter, body=[], orelse=[]), ast_node)
    elif isinstance(ast_node, ast.Try):
        ast_node = ast.copy_location(ast.Try(body=[], handlers=[], orelse=[], finalbody=[]), ast_node)
    elif isinstance(ast_node, ast.With):
        ast_node = ast.copy_location(ast.With(items=ast_node.items, body=[]), ast_node)
    src_code = astor.to_source(ast_node).strip().replace("\n", "")
    src_code += f":Line {node.lineno}\n"
    return src_code



def visit_blocks(graph, block, visited, calls):
    if block.id in visited:
        return

    visited.append(block.id)

    # Get source code from block statements
    src_code = ''
    for stmt in block.statements:
        src_code += to_src_without_children(stmt)

    # Add source code to block label
    label = '{}\n{}'.format(block, src_code)

    graph.node(str(block.id), label)

    for exit in block.exits:
        if calls or exit.name not in block.cfg.functioncfgs:
            visit_blocks(graph, exit.target, visited, calls)
            graph.edge(str(block.id), str(exit.target.id),
                       label=str(exit.exitcase) if exit.exitcase else None)
    # Get source code from block statements


def get_all_block_of_cfgs(flattened_cfg):
    all_block = []
    for fqn, cfg in flattened_cfg.items():
        all_block+=cfg.get_all_blocks()

    return all_block

def get_all_block_of_cfgs_dict(all_block):
        return {block.id: block for block in all_block}


def build_and_draw_cfgs(flattened_cfg, graph):
    for fqn, cfg in flattened_cfg.items():
        with graph.subgraph(name='cluster_'+fqn.replace(".", "_")) as sub:
            sub.attr(label=fqn)
            visit_blocks(sub, cfg.entryblock, visited=[], calls=True)



def reverse_depth_first_search(start_block,end_block,path=None, paths=None, visited=None):
    global def_dict,class_dict,flattened_cfg
    if paths is None:
        paths = []

    if path is None:
        path = [start_block]
    else:
        path = [start_block] + path

    if visited is None:
        visited = {}

    visited[start_block] = visited.get(start_block, 0) + 1  # increment visit count for nodes

    if len(start_block.predecessors) == 0 or all(visited.get(link.source, 0) == 1 and (len(link.source.predecessors) != 2 ) for link in start_block.predecessors) or end_block ==start_block:
        reversed_path = path[::-1]
        paths.append(reversed_path)
        visited[start_block] = visited.get(start_block, 0) - 1
        print('Path:', [block.id for block in path])
        return paths

    for link in start_block.predecessors:
        if link.source not in visited or visited[link.source] == 0 or(visited[link.source] == 1 and  len(link.source.predecessors) >= 2):
            reverse_depth_first_search(link.source,end_block, path, paths, visited)
        elif (visited[link.source] == 1 and len(link.source.predecessors) == 1):
            reversed_path = path[::-1]
            paths.append(reversed_path)
            print('Path:', [block.id for block in path])
    visited[start_block] = visited.get(start_block, 0) - 1
    return paths




def get_all_code_block_index(all_block):
    all_code_block_index_dict = {}
    for block in all_block.values():
        for index,node in enumerate(block.statements):
            node_string = to_src_without_children(node)
            node_string_lineno = re.search(r':Line (\d+)', node_string)
            node_string_lineno = node_string_lineno.group(1) if node_string_lineno else None
            all_code_block_index_dict[node_string_lineno] = (block.id,index)
    return all_code_block_index_dict


def depth_first_search(start_block,end_block,path=None, paths=None, visited=None):
    if paths is None:
        paths = []

    if path is None:
        path = [start_block]
    else:
        path = [start_block] + path

    if visited is None:
        visited = {}

    visited[start_block] = visited.get(start_block, 0) + 1  # increment visit count for nodes

    if len(start_block.exits) == 0 or all(visited.get(link.target, 0) == 1 and (len(link.target.exits) != 2 ) for link in start_block.exits) or end_block ==start_block:
        reversed_path = path[::-1]
        paths.append(reversed_path)
        print('Path:', [block.id for block in path])
        visited[start_block] = visited.get(start_block, 0) - 1
        return paths

    for link in start_block.exits:
        if link.target not in visited or visited[link.target] ==0 or (visited[link.target] == 1 and len(link.target.exits) >= 2):
            depth_first_search(link.target,end_block, path, paths, visited)
        elif (visited[link.target] == 1 and len(link.target.exits) == 1):
            reversed_path = path[::-1]
            paths.append(reversed_path)
            print('Path:', [block.id for block in path])
    visited[start_block] = visited.get(start_block, 0) - 1
    return paths






def find_in_block(node_string):
    node_string_lineno = re.search(r':Line (\d+)', node_string)
    node_string_lineno = node_string_lineno.group(1) if node_string_lineno else None
    for mod in flattened_cfg:
        cfgs = flattened_cfg[mod]
        visited = None
        if visited is None:
            visited = set()

        # start from the entry block
        stack = [cfgs.entryblock]

        while stack:
            find = False
            find_index =-1
            block = stack.pop()

            if block in visited:
                continue
            for i, stmt in enumerate(block.statements):
                stmt_string = to_src_without_children(stmt)
                stmt_string_lineno = re.search(r':Line (\d+)', stmt_string)
                stmt_string_lineno = stmt_string_lineno.group(1) if stmt_string_lineno else None
                if stmt_string_lineno == node_string_lineno:
                    find = True
                    find_index = i
                    break
            if find:
                return block,find_index

            visited.add(block)

            # print or do something with block

            for exit in block.exits:
                if exit.target not in visited:
                    stack.append(exit.target)
    raise ValueError(f"The value '{node_string}' could not be found in the list.")


def get_def_class_info(flattened_cfg):
    def_dict = {}
    class_dict = {}
    def_cfg_dict={}
    for key in flattened_cfg:
        cfg= flattened_cfg[key]
        if key == 'mod':
            for node in cfg.entryblock.statements:
                if isinstance(node, ast.ClassDef):
                    class_dict[node.name] = {'base': [],
                                             'is_init': False,
                                             'func': []}
                    for base in node.bases:
                        class_dict[node.name]['base'].append(astor.to_source(base))
                    for func in node.body:
                        if isinstance(func, ast.FunctionDef):
                            class_dict[node.name]['func'].append(func.name)

                    if '__init__' in class_dict[node.name]['func']:
                        class_dict[node.name]['is_init'] = True

        if len(flattened_cfg[key].function_args) == 0:
            name = key.replace('mod.', '')
            if '_init_' in name.lower():
                name = name.replace(".__init__", '')
            def_dict[name] = cfg
            def_cfg_dict[name] =key
    return  def_dict, class_dict ,def_cfg_dict

def read_python_file(filename):
    with open(filename, 'r') as file:
        return file.read()



def get_method(class_name, method_name, inheritance_map, class_dict):

    for cls in c3_linearization(class_name, inheritance_map):
        class_node = find_class_definition(cls, class_dict)

        for func_name in class_node['func']:
            if  func_name == method_name:
                return cls+'.'+func_name

    raise Exception(f"The method {method_name} was not found in class {class_name} or its parent classes.")

def find_class_definition(class_name, class_dict):

    for name in class_dict:
        if name == class_name:
            return class_dict[name]

    raise Exception(f"The class {class_name} is not defined or the class definition for {class_name} cannot be found.")

def c3_linearization(cls, inheritance_map):
    if not inheritance_map[cls]:
        return [cls]
    else:
        parents = inheritance_map[cls]
        return merge([[cls]] + [c3_linearization(parent, inheritance_map) for parent in parents] + [parents])

def merge(seqs):
    res = []
    while seqs:
        non_empty_seqs = [s for s in seqs if s]  # Remove empty sequences
        if not non_empty_seqs:
            return res
        for s in non_empty_seqs:  # Find merge candidates among seq heads
            candidate = s[0]
            if not any(candidate in s[1:] for s in non_empty_seqs):
                break  # Good candidate
        else:
            raise Exception("Inconsistent hierarchy, no good candidate found")
        res.append(candidate)
        for seq in non_empty_seqs:
            if seq[0] == candidate:
                del seq[0]  # Remove the chosen candidate
        seqs = [s for s in seqs if s]  # Remove empty sequences
    return res


def DFG_reverse_depth_first_search(start_param, path=None, paths=None):
    global def_dict, const_dict, const_cfg_dict, all_code_block_index_dict, ssa_results, def_cfg_dict

    if paths is None:
        paths = []

    if path is None:
        path = [start_param]
    else:
        path = path + [start_param]

    current_cfg = const_cfg_dict[start_param]  # get start_param's CFG
    current_const_code = const_dict[current_cfg][start_param]  # get start_param's initialization statement
    current_const_lineno = current_const_code.lineno  # get line number of initialization statement
    (next_block_id, next_index) = all_code_block_index_dict[str(current_const_lineno)]
    next_SSA = ssa_results[current_cfg][next_block_id][next_index]

    start_params = []
    for param_need_trace in next_SSA:
        if len(next_SSA[param_need_trace]) == 0:
            continue
        for possible_value_id in next_SSA[param_need_trace]:
            start_params.append((param_need_trace, possible_value_id))

    if start_params:
        for start_param in start_params:
            # Avoid cycles by not revisiting nodes that are already in the current path
            if start_param not in path:
                DFG_reverse_depth_first_search(start_param, path, paths)
    else:
        paths.append(path)

    return paths

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError("Cannot convert string to boolean")




class coustomChecker(BaseChecker):
    global flattened_cfg
    name = "coustom_Checker"
    priority = -1
    msgs = {
    }
    def __init__(self,path, linter=None):
        super().__init__(linter)
        with open(path, "r", encoding="utf-8") as json_file:
            function_params = json.load(json_file)
        message_code = function_params.get("MessageCode")
        message_detail = function_params.get(message_code)

        if message_detail:
            message_detail_tuple = (message_detail['Message'], message_detail['Code'], message_detail['Description'])
            message_dict = {message_code: message_detail_tuple}
        else:
            print("未找到对应的错误代码")
        self.msgs[message_code] = (message_detail['Message'], message_detail['Code'], message_detail['Description'])
        self.Code = str(message_detail['Code'])

        self.classdef_lineno_scope = function_params.get("classdef_lineno_scope")
        self.start_sign = function_params.get("start_sign")
        self.start_sign_necessary = str_to_bool(function_params.get("start_sign_necessary"))
        self.fg_type =  function_params.get("fg_type")
        self.keyword_need_trace =  function_params.get("keyword_need_trace")
        self.arg_need_trace_pos =  function_params.get("arg_need_trace_pos")
        self.end_sign_necessary = str_to_bool(function_params.get("end_sign_necessary"))
        self.end_sign =  function_params.get("end_sign")

        self.use_common_api_name =function_params.get("use_common_api_name")



        self.find_sign =  function_params.get("find_sign")
        for commonly_API_names_dict_key in self.use_common_api_name:
            if  commonly_API_names_dict_key in commonly_API_names_dict:
                self.find_sign+= commonly_API_names_dict[commonly_API_names_dict_key]
        self.find_all_or_any =  function_params.get("find_all_or_any")
        self.trace_type =  function_params.get("trace_type")
        self.Find_the_basis_to_be_correct_or_incorrect =  str_to_bool(function_params.get("Find_the_basis_to_be_correct_or_incorrect"))

    def visit_classdef(self, node):
        self.classdef_lineno_scope[node.name] = (node.fromlineno, node.end_lineno)

    def Custom_criteria_for_judgment(self,current_node):
        if isinstance(current_node, astroid.node_classes.NodeNG):
            if not (isinstance(current_node,astroid.ClassDef) or isinstance(current_node,astroid.FunctionDef)):
                if self.find_all_or_any == "any":
                    if any(find_sign in current_node.as_string() for find_sign in self.find_sign):
                        return  True
                elif self.find_all_or_any == "all":
                    curr_code=current_node.as_string()
                    for find_sign in self.find_sign:
                        index = -1
                        if not find_sign in curr_code:
                            return False
                        index = curr_code.find(find_sign)
                        if index != -1:
                            curr_code=curr_code[:index] + curr_code[index + len(find_sign):]
                    return True
                else:
                    raise Exception("find_all_or_any should be either 'all' or 'any'")

            else:
                if self.find_all_or_any == "any":
                    if any(find_sign in to_src_without_children_astroid(current_node) for find_sign in self.find_sign):
                        return True
                elif self.find_all_or_any == "all":
                    curr_code = to_src_without_children_astroid(current_node)
                    for find_sign in self.find_sign:
                        index = -1
                        if not find_sign in curr_code:
                            return False
                        index = curr_code.find(find_sign)
                        if index != -1:
                            curr_code = curr_code[:index] + curr_code[index + len(find_sign):]
                    return True

                else:
                    raise Exception("find_all_or_any should be either 'all' or 'any'")
            return False
        elif isinstance(current_node,str):
            if self.find_all_or_any == "any":
                if any(find_sign in current_node for find_sign in self.find_sign):
                    return  True
            elif self.find_all_or_any == "all":

                curr_code = current_node
                for find_sign in self.find_sign:
                    index = -1
                    if not find_sign in curr_code:
                        return False
                    index = curr_code.find(find_sign)
                    if index != -1:
                        curr_code = curr_code[:index] + curr_code[index + len(find_sign):]
                return True

            else:
                raise Exception("find_all_or_any should be either 'all' or 'any'")
            return False

        if not (isinstance(current_node, ast.ClassDef) or isinstance(current_node, ast.FunctionDef)):
            if self.find_all_or_any == "any":
                if any(find_sign in astor.to_source(current_node) for find_sign in self.find_sign):
                    return True
            elif self.find_all_or_any == "all":
                curr_code = astor.to_source(current_node)
                for find_sign in self.find_sign:
                    index = -1
                    if not find_sign in curr_code:
                        return False
                    index = curr_code.find(find_sign)
                    if index != -1:
                        curr_code = curr_code[:index] + curr_code[index + len(find_sign):]
                return True

            else:
                raise Exception("find_all_or_any should be either 'all' or 'any'")
            return False
        else:
            if self.find_all_or_any == "any":
                if any(find_sign in to_src_without_children(current_node) for find_sign in self.find_sign):
                    return True
            elif self.find_all_or_any == "all":
                curr_code = to_src_without_children(current_node)
                for find_sign in self.find_sign:
                    index = -1
                    if not find_sign in curr_code:
                        return False
                    index = curr_code.find(find_sign)
                    if index != -1:
                        curr_code = curr_code[:index] + curr_code[index + len(find_sign):]
                return True

            else:
                raise Exception("find_all_or_any should be either 'all' or 'any'")
        return False



    def CFG_Forward_traversal(self,block,level,index_orgin = None,end_block = None,end_index= None):
        global isadd
        is_error = self.Find_the_basis_to_be_correct_or_incorrect
        paths = depth_first_search(block,end_block)

        for path in paths:
            is_error = self.Find_the_basis_to_be_correct_or_incorrect
            index = index_orgin
            for id in range(len(path)):
                if id == 0 and level == 0 and self.trace_type!="all_code":

                    if isinstance(path[id].statements[index], ast.ClassDef):
                        for class_func in path[id].statements[index].body:
                            if isinstance(class_func, ast.FunctionDef):
                                if class_func.name != "__init__":
                                    is_error_class = self.CFG_Forward_traversal(
                                        def_dict[path[id].statements[index].name+"."+class_func.name].entryblock, level=level + 1,
                                        index_orgin=0)
                                else:
                                    is_error_class = self.CFG_Forward_traversal(
                                        def_dict[path[id].statements[index].name].entryblock,
                                        level=level + 1,
                                        index_orgin=0)
                                if is_error_class != self.Find_the_basis_to_be_correct_or_incorrect:
                                    is_error = is_error_class
                        if level == 0 and is_error:
                            node = ast_to_astroid(path[0].statements[0]).body[0]

                            isadd = True
                            self.add_message( self.Code, line=path[0].statements[0].lineno, node=node)
                            return None
                    elif isinstance(path[id].statements[index], ast.FunctionDef):
                        is_error = self.CFG_Forward_traversal(
                            def_dict[path[id].statements[index].name].entryblock,
                            level=level + 1,
                            index_orgin=0)
                        if level == 0 and is_error:
                            node = ast_to_astroid(path[0].statements[0]).body[0]
                            isadd = True
                            self.add_message( self.Code, line=path[0].statements[0].lineno, node=node)
                            return None
                    if index == len(path[id].statements) - 1:
                        continue
                    index = index + 1
                else:
                    index = 0

                for_num = len(path[id].statements)
                index_copy = index
                for i in range(for_num-index_copy):
                    current_node = path[id].statements[index]
                    if self.Custom_criteria_for_judgment(current_node):
                        is_error = not self.Find_the_basis_to_be_correct_or_incorrect

                        if self.trace_type == "all_code":
                            if is_error:
                                node = ast_to_astroid(path[0].statements[0]).body[0]
                                isadd = True
                                self.add_message(self.Code, line=path[0].statements[0].lineno, node=node)
                            return None
                    current_node_source = to_src_without_children(current_node)
                    current_lineno = re.search(r':Line (\d+)', current_node_source)
                    current_lineno = int(current_lineno.group(1) if current_lineno else None)
                    current_node_source = re.sub(r':Line \d+', '', current_node_source)
                    if "self." in current_node_source:
                        for key_class in self.classdef_lineno_scope:
                            (a,b)=self.classdef_lineno_scope[key_class]
                            if a<=current_lineno and b>=current_lineno:
                                current_node_source.replace("self.",key_class+".")
                                break
                    for key in def_dict:
                        if key + '(' in current_node_source and (not isinstance(current_node,astroid.FunctionDef))\
                                and (not isinstance(current_node,ast.FunctionDef)) and (not isinstance(current_node,astroid.ClassDef))\
                                and (not isinstance(current_node,ast.ClassDef)):
                            if def_dict[key].entryblock != block:
                                if not(key in class_dict and (".Module" in str(class_dict[key]["base"]))):
                                    is_error = self.CFG_Forward_traversal(def_dict[key].entryblock, level=level + 1,
                                                              index_orgin=0)
                                else:
                                    is_error_1 =None

                                    if (key + ".forward") in def_dict:
                                        is_error_1 = self.CFG_Forward_traversal(def_dict[key+".forward"].entryblock, level=level + 1,
                                                          index_orgin=0)
                                    is_error_2 = self.CFG_Forward_traversal(def_dict[key].entryblock, level=level + 1,
                                                              index_orgin=0)
                                    if is_error_1 and is_error_1 != self.Find_the_basis_to_be_correct_or_incorrect:
                                        is_error = is_error_1
                                    elif is_error_2!= self.Find_the_basis_to_be_correct_or_incorrect:
                                        is_error = is_error_2
                    index = index + 1
                    if  is_error != self.Find_the_basis_to_be_correct_or_incorrect:
                        break
                    if path[id] == end_block and index >= end_index:
                        break

                if  is_error != self.Find_the_basis_to_be_correct_or_incorrect:
                    break
                if path[id] == end_block and index >= end_index:
                    break

            if level != 0 and is_error :
                return is_error

            if level == 0 and is_error:
                node = ast_to_astroid(path[0].statements[0]).body[0]

                isadd = True
                self.add_message( self.Code, line=path[0].statements[0].lineno, node=node)
        return is_error

    def CFG_Reverse_tracing(self,block,level,index_orgin = None,end_block = None,end_index= None):
        global isadd
        is_error = self.Find_the_basis_to_be_correct_or_incorrect
        paths = reverse_depth_first_search(block,end_block)

        for path in paths:
            is_error = self.Find_the_basis_to_be_correct_or_incorrect
            index = index_orgin
            for id in range(len(path)):
                if id == 0 and level == 0:
                    if isinstance(path[id].statements[index], ast.ClassDef):
                        for class_func in path[id].statements[index].body:
                            if isinstance(class_func, ast.FunctionDef):
                                if class_func.name != "__init__":
                                    is_error_class = self.CFG_Forward_traversal(
                                        def_dict[path[id].statements[index].name + "." + class_func.name].entryblock,
                                        level=level + 1,
                                        index_orgin=0)
                                else:
                                    is_error_class = self.CFG_Forward_traversal(
                                        def_dict[path[id].statements[index].name].entryblock,
                                        level=level + 1,
                                        index_orgin=0)
                                if is_error_class != self.Find_the_basis_to_be_correct_or_incorrect:
                                    is_error = is_error_class
                            if level == 0 and is_error:
                                node = ast_to_astroid(path[0].statements[0]).body[0]

                                isadd = True
                                self.add_message( self.Code, line=path[0].statements[0].lineno, node=node)
                                return None
                    elif isinstance(path[id].statements[index], ast.FunctionDef):
                        is_error = self.CFG_Forward_traversal(
                            def_dict[path[id].statements[index].name].entryblock,
                            level=level + 1,
                            index_orgin=0)
                        if level == 0 and is_error:
                            node = ast_to_astroid(path[0].statements[0]).body[0]

                            isadd = True
                            self.add_message( self.Code, line=path[0].statements[0].lineno, node=node)
                            return None

                    if index == 0:
                        continue
                    index = index - 1
                else:
                    index = len(path[id].statements) - 1

                for_num = index +1
                for i in range(for_num):
                    current_node = path[id].statements[index]
                    if self.Custom_criteria_for_judgment(current_node):
                        is_error = not self.Find_the_basis_to_be_correct_or_incorrect
                        if self.trace_type == "all_code":
                            if is_error:
                                node = ast_to_astroid(path[0].statements[0]).body[0]
                                isadd = True
                                self.add_message(self.Code, line=path[0].statements[0].lineno, node=node)
                            return None
                    current_node_source = to_src_without_children(current_node)
                    current_lineno = re.search(r':Line (\d+)', current_node_source)
                    current_lineno = int(current_lineno.group(1) if current_lineno else None)
                    current_node_source = re.sub(r':Line \d+', '', current_node_source)
                    if "self." in current_node_source:
                        for key_class in self.classdef_lineno_scope:
                            (a,b)=self.classdef_lineno_scope[key_class]
                            if a <= current_lineno and b >=current_lineno:
                                current_node_source=current_node_source.replace("self.",key_class+".")
                                break
                    for key in def_dict:
                        if key + '(' in current_node_source:
                            if def_dict[key].entryblock != block:
                                if not (key in class_dict and (".Module" in str(class_dict[key]["base"]))):
                                    is_error = self.CFG_Forward_traversal(def_dict[key].entryblock, level=level + 1,
                                                                          index_orgin=0)
                                else:
                                    is_error_1 = None
                                    if (key + ".forward")in def_dict:
                                        is_error_1 = self.CFG_Forward_traversal(def_dict[key + ".forward"].entryblock,
                                                                            level=level + 1,
                                                                            index_orgin=0)
                                    is_error_2 = self.CFG_Forward_traversal(def_dict[key].entryblock, level=level + 1,
                                                                            index_orgin=0)
                                    if is_error_1 and is_error_1 != self.Find_the_basis_to_be_correct_or_incorrect:
                                        is_error = is_error_1
                                    elif is_error_2 != self.Find_the_basis_to_be_correct_or_incorrect:
                                        is_error = is_error_2
                    index = index - 1
                    if is_error != self.Find_the_basis_to_be_correct_or_incorrect:
                        break

                    if path[id] == end_block and index >= end_index:
                        break

                if is_error != self.Find_the_basis_to_be_correct_or_incorrect:
                    break

                if path[id] == end_block and index >= end_index:
                    break

            if level != 0 and is_error:
                return is_error

            if level == 0 and is_error:
                node = ast_to_astroid(path[0].statements[0]).body[0]

                isadd = True
                self.add_message( self.Code, line=path[0].statements[0].lineno, node=node)
        return is_error

    def DFG_Reverse_tracing(self,block,level,param_need_trace,index = None):
        global isadd
        global ssa_results,const_dict
        block_id=block.id

        for cfg_name in ssa_results.keys():
            for key in ssa_results[cfg_name].keys():
                if key == block_id:
                    start_params=[]
                    if param_need_trace in  ssa_results[cfg_name][key][index]:
                        for possible_value_id in ssa_results[cfg_name][key][index][param_need_trace] :
                            start_params.append((param_need_trace,possible_value_id))
                        break
        paths=[]
        for start_param in start_params:
            paths+=DFG_reverse_depth_first_search(start_param)



        if len(paths) ==0:
            is_error = self.Find_the_basis_to_be_correct_or_incorrect
            if self.Custom_criteria_for_judgment(param_need_trace):
                is_error = not self.Find_the_basis_to_be_correct_or_incorrect
                if self.trace_type == "all_code":
                    return None
            if level == 0 and is_error:

                isadd = True
                self.add_message( self.Code, line=block.statements[0].lineno, node=ast_to_astroid(block.statements[0]))

                return None


        for path in paths:
            is_error = self.Find_the_basis_to_be_correct_or_incorrect
            error_node = None
            for param in path:
                current_cfg_name=const_cfg_dict[param]
                current_node=const_dict[current_cfg_name][param]
                if self.Custom_criteria_for_judgment(current_node):
                    is_error = not self.Find_the_basis_to_be_correct_or_incorrect
                    if self.trace_type=="all_code":
                        return None
                current_node_source = to_src_without_children(current_node)
                current_lineno = re.search(r':Line (\d+)', current_node_source)
                current_lineno = int(current_lineno.group(1) if current_lineno else None)
                current_node_source = re.sub(r':Line \d+', '', current_node_source)
                if "self." in current_node_source:
                    for key_class in self.classdef_lineno_scope:
                        (a, b) = self.classdef_lineno_scope[key_class]
                        if a <= current_lineno and b >= current_lineno:
                            current_node_source = current_node_source.replace("self.", key_class + ".")
                            break
                for key in def_dict:
                    if key + '(' in current_node_source:
                        current_cfg_name_call = def_cfg_dict[key]
                        return_node = flattened_cfg[current_cfg_name_call].entryblock.statements[-1]
                        if isinstance(return_node, ast.Return):
                            return_node_lineno = return_node.lineno
                            (next_block_id, next_index) = all_code_block_index_dict[str(return_node_lineno)]
                            return_value_node = return_node.value
                            if isinstance(return_value_node, ast.Tuple):
                                # If it is, print each element of the tuple
                                return_values = [ast.unparse(elt) for elt in return_value_node.elts]
                                start_params_call = return_values  # Outputs: ['c', 'd']
                            else:
                                start_params_call = [ast.unparse(return_value_node)]
                            for start_param_call in start_params_call:
                                is_error = self.DFG_Reverse_tracing(block=all_block_dict[next_block_id],level = level+1, param_need_trace =start_param_call ,index= next_index)
                        else:
                            raise Exception("Error: The assignment statement references a function that does not return a value, and the data flow trace terminates. Please check the code.")
                if  is_error :
                    error_node=current_node
                    break
            if level != 0 and is_error:
                return is_error

            if level == 0 and is_error:

                isadd = True
                self.add_message( self.Code, line=error_node.lineno, node=ast_to_astroid(error_node))

    def recursive_search(self, curr_node,sign):
        for child in curr_node.get_children():

            if child.as_string().count('\n') <= 1 and (str(sign) in str(child.as_string())):
                return child
            elif child.as_string().count('\n') >= 1 :
                result = self.recursive_search(child,sign)
                if result is not None:
                    return result
        return None

    def visit_module(self, node):
        global isadd
        if len(self.start_sign)>0:
            start_node =None
            for curr_node in node.body:
                matching_sign = next(
                    (start_sign for start_sign in self.start_sign if start_sign in curr_node.as_string()), None)
                if matching_sign:
                    if self.fg_type == "dfg":
                        # for child in curr_node.get_children():
                        start_node =  self.recursive_search(curr_node,matching_sign)

                        if start_node ==None:
                            start_node =curr_node
                    elif self.fg_type == "cfg":

                        start_node = self.recursive_search(curr_node, matching_sign)
                        if start_node ==None:
                            start_node =curr_node
                    else:

                        raise ValueError("Unknown flow graph Type")
            if not start_node:
                if self.start_sign_necessary:
                    global isfind
                    isfind = False
                    print("None of"+str(self.start_sign) + "in your code,please check your code,If you are using multiple subcheckers, please ignore this message")
                return None
            start_node_string = to_src_without_children_astroid(start_node)
            block, index = find_in_block(start_node_string)
            index = int(index)
            block_end = None
            index_end = None

            if len(self.end_sign)>0:
                end_node = None
                for curr_node in node.body:
                    matching_sign_end = next(
                        (end_sign for end_sign in self.end_sign if end_sign in curr_node.as_string()), None)
                    if matching_sign_end:
                        if self.fg_type == "dfg":
                            end_node = self.recursive_search(curr_node, matching_sign_end)
                            if end_node == None:
                                end_node = curr_node
                        elif self.fg_type == "cfg":
                            end_node = self.recursive_search(curr_node, matching_sign_end)
                            if end_node == None:
                                end_node = curr_node
                        else:
                            raise ValueError("Unknown flow graph Type")

                    if end_node and self.trace_type == "reverse" and end_node.lineno < start_node.lineno:
                        end_node_string = to_src_without_children_astroid(end_node)
                        block_end, index_end = find_in_block(end_node_string)
                        index_end = int(index_end)
                    elif end_node and self.trace_type == "forward" and end_node.lineno > start_node.lineno:
                        end_node_string = to_src_without_children_astroid(end_node)
                        block_end, index_end = find_in_block(end_node_string)
                        index_end = int(index_end)

            if self.fg_type == "dfg":
                if self.trace_type == "forward":
                    print("There is no 'forward' method in dfg. Please check your settings for errors. The processing will automatically proceed in reverse")
                params_need_trace = []
                if not isinstance(start_node, astroid.Call):
                    for child in start_node.get_children():
                        if isinstance(child, astroid.Call):
                            start_node = child
                            break
                for pos in self.arg_need_trace_pos:
                    max_pos = len(start_node.args) - 1
                    if pos > max_pos:

                        isadd = True
                        self.add_message( self.Code, line=start_node.lineno, node=start_node)
                        return None
                    if isinstance(start_node.args[pos], astroid.Const):
                        continue
                    params_need_trace += [start_node.args[pos].as_string()]
                for key in self.keyword_need_trace:
                    is_in = False
                    for key_node in start_node.keywords:
                        if key_node.arg == key:
                            is_in =True
                            if self.Custom_criteria_for_judgment(key_node):
                                if not self.Find_the_basis_to_be_correct_or_incorrect:
                                    isadd = True
                                    self.add_message( self.Code, line=start_node.lineno, node=start_node)
                                return None
                            params_need_trace += [key_node.value.as_string()]
                    if not is_in :

                        isadd = True
                        self.add_message( self.Code, line=start_node.lineno, node=start_node)
                for param_need_trace in params_need_trace:
                    self.DFG_Reverse_tracing(block=block, level=0, param_need_trace=param_need_trace, index=index)
            elif self.fg_type == "cfg":
                if self.trace_type == "reverse":
                    self.CFG_Reverse_tracing(block=block,level = 0, index_orgin= index,end_block = block_end,end_index=index_end)
                elif self.trace_type == "forward":
                    self.CFG_Forward_traversal(block=block, level=0, index_orgin=index,end_block = block_end,end_index=index_end)
        elif self.trace_type == "all_code":
            self.CFG_Forward_traversal(block=all_block_dict[1], level=0, index_orgin=0)















def register(linter):
    linter.register_checker(coustomChecker(path,linter))





def main(path_user,test_scripts):
    global path,all_block_dict,def_dict,class_dict,def_cfg_dict,code,all_code_block_index_dict,flattened_cfg,isfind,isadd,ssa_results,const_dict,const_cfg_dict
    path = path_user
    isfind =True
    isadd = False


    for script_path in test_scripts:
        test_script = os.path.basename(script_path)
        cfg_builder = CFGBuilder()
        flattened_cfg = cfg_builder.build_from_file(test_script ,
                                                    script_path,
                                                    flattened=True)

        all_block = get_all_block_of_cfgs(flattened_cfg)
        all_block_dict= get_all_block_of_cfgs_dict(all_block)
        all_code_block_index_dict=get_all_code_block_index(all_block_dict)
        m_ssa = SSA()
        ssa_results={}
        const_dict ={}
        const_cfg_dict ={}
        for key in flattened_cfg:
            ssa_results_temp, const_dict_temp = m_ssa.compute_SSA(flattened_cfg[key])
            ssa_results[key]= ssa_results_temp
            const_dict[key]= const_dict_temp
            for a in const_dict_temp:
                const_cfg_dict[a]=key

        graph = gv.Digraph(name='cluster_main', format='png', graph_attr={'label': 'main'})
        def_dict,class_dict,def_cfg_dict = get_def_class_info(flattened_cfg)
        code = read_python_file(script_path)

        build_and_draw_cfgs(flattened_cfg, graph)
        graph.render('flattened_cfg', view=False)
        print(f"Currently checking the script ：{script_path}")
        Run([
            script_path,
            "--load-plugins", "core.core",
            "--disable=all",
            "--enable=coustom_Checker"
        ], do_exit=False)
        return isadd,isfind



