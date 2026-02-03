import graphviz
import re

def visualize_computational_graph(raw_output):
    # Initialize the graph
    dot = graphviz.Digraph(comment='Autograd Graph', graph_attr={'rankdir': 'BT'}) # Bottom-up flow

    # Regex to capture the C struct output
    # Matches: Address, Data, Grad, isLeaf, Child1, Child2, Op
    pattern = r"Value (0x[0-9a-fA-F]+)\s+Data: ([\d\.-]+)\s+Grad: ([\d\.-]+)\s+isLeaf: (\d)\s+Child1 Pointer: (0x[0-9a-fA-F]+)\s+Child2 Pointer: (0x[0-9a-fA-F]+)\s+Operation:\s*(.)?"
    
    matches = re.findall(pattern, raw_output)
    
    seen_nodes = set()

    for addr, data, grad, is_leaf, c1, c2, op in matches:
        if addr in seen_nodes:
            continue
        seen_nodes.add(addr)

        # Formatting the label to show EVERYTHING
        op_label = op.strip() if op.strip() else "Leaf"
        node_label = f"{{ ADDR: {addr} | OP: {op_label} | DATA: {data} | GRAD: {grad} }}"
        
        # Color: Green for leaves (inputs), Blue for operations
        color = '#e1f5fe' if is_leaf == '1' else '#fff9c4'
        
        dot.node(addr, label=node_label, shape='record', style='filled', fillcolor=color)

        # Create edges to children (if they aren't NULL/0x0)
        if c1 != '0x0' and c1 != '(nil)':
            dot.edge(addr, c1, label="left")
        if c2 != '0x0' and c2 != '(nil)':
            dot.edge(addr, c2, label="right")

    return dot
# Paste your raw string here
data_input = """Value 0x100b3b5a0
Data: 46.128490
Grad: 1.000000
isLeaf: 0
Child1 Pointer: 0x100b3b570
Child2 Pointer: 0x100b3b570
Operation: *

Value 0x100b3b570
Data: -6.791796
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x100b3b510
Child2 Pointer: 0x100b3b540
Operation: +

Value 0x100b3b510
Data: 1.208204
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x100b3b4b0
Child2 Pointer: 0x100b3b4e0
Operation: +

Value 0x100b3b4b0
Data: 0.456012
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x100b3b450
Child2 Pointer: 0x100b3b480
Operation: +

Value 0x100b3b450
Data: 0.153512
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x100b3b3d0
Child2 Pointer: 0x100b3b420
Operation: +

Value 0x100b3b3d0
Data: 0.000000
Grad: -13.583591
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b420
Data: 0.153512
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x965008ab0
Child2 Pointer: 0x100b3afd0
Operation: *

Value 0x965008ab0
Data: 0.310000
Grad: -6.726595
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3afd0
Data: 0.495200
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3af70
Child2 Pointer: 0x100b3afa0
Operation: +

Value 0x100b3af70
Data: 0.106400
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3af10
Child2 Pointer: 0x100b3af40
Operation: +

Value 0x100b3af10
Data: 0.056000
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3ae90
Child2 Pointer: 0x100b3aee0
Operation: +

Value 0x100b3ae90
Data: 0.000000
Grad: -11.292771
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3aee0
Data: 0.056000
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008900
Child2 Pointer: 0x100b3a560
Operation: *

Value 0x965008900
Data: 0.160000
Grad: -3.952470
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a560
Data: 0.350000
Grad: -1.806843
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3af40
Data: 0.050400
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008930
Child2 Pointer: 0x100b3a590
Operation: *

Value 0x965008930
Data: 0.240000
Grad: -2.371482
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a590
Data: 0.210000
Grad: -2.710265
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3afa0
Data: 0.388800
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008960
Child2 Pointer: 0x100b3a5c0
Operation: *

Value 0x965008960
Data: 0.720000
Grad: -6.098097
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a5c0
Data: 0.540000
Grad: -8.130795
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b480
Data: 0.302500
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x965008ae0
Child2 Pointer: 0x100b3b1b0
Operation: *

Value 0x965008ae0
Data: 0.660000
Grad: -6.225820
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b1b0
Data: 0.458334
Grad: -8.965171
isLeaf: 0
Child1 Pointer: 0x100b3afd0
Child2 Pointer: 0x0
Operation: t

Value 0x100b3afd0
Data: 0.495200
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3af70
Child2 Pointer: 0x100b3afa0
Operation: +

Value 0x100b3af70
Data: 0.106400
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3af10
Child2 Pointer: 0x100b3af40
Operation: +

Value 0x100b3af10
Data: 0.056000
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3ae90
Child2 Pointer: 0x100b3aee0
Operation: +

Value 0x100b3ae90
Data: 0.000000
Grad: -11.292771
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3aee0
Data: 0.056000
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008900
Child2 Pointer: 0x100b3a560
Operation: *

Value 0x965008900
Data: 0.160000
Grad: -3.952470
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a560
Data: 0.350000
Grad: -1.806843
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3af40
Data: 0.050400
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008930
Child2 Pointer: 0x100b3a590
Operation: *

Value 0x965008930
Data: 0.240000
Grad: -2.371482
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a590
Data: 0.210000
Grad: -2.710265
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3afa0
Data: 0.388800
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008960
Child2 Pointer: 0x100b3a5c0
Operation: *

Value 0x965008960
Data: 0.720000
Grad: -6.098097
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a5c0
Data: 0.540000
Grad: -8.130795
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b4e0
Data: 0.752192
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x965008b10
Child2 Pointer: 0x100b3b360
Operation: *

Value 0x965008b10
Data: 0.460000
Grad: -22.211887
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b360
Data: 1.635200
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x100b3b300
Child2 Pointer: 0x100b3b330
Operation: +

Value 0x100b3b300
Data: 1.243200
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x100b3b2a0
Child2 Pointer: 0x100b3b2d0
Operation: +

Value 0x100b3b2a0
Data: 0.960000
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x100b3b040
Child2 Pointer: 0x100b3b270
Operation: +

Value 0x100b3b040
Data: 0.000000
Grad: -6.248452
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b270
Data: 0.960000
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x965008990
Child2 Pointer: 0x100b3a680
Operation: *

Value 0x965008990
Data: 0.960000
Grad: -6.248452
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a680
Data: 1.000000
Grad: -5.998514
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b2d0
Data: 0.283200
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x9650089c0
Child2 Pointer: 0x100b3a6b0
Operation: *

Value 0x9650089c0
Data: 0.590000
Grad: -2.999257
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a6b0
Data: 0.480000
Grad: -3.686587
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b330
Data: 0.392000
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x9650089f0
Child2 Pointer: 0x100b3a6e0
Operation: *

Value 0x9650089f0
Data: 0.700000
Grad: -3.499133
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a6e0
Data: 0.560000
Grad: -4.373917
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b540
Data: -8.000000
Grad: -13.583591
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b570
Data: -6.791796
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x100b3b510
Child2 Pointer: 0x100b3b540
Operation: +

Value 0x100b3b510
Data: 1.208204
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x100b3b4b0
Child2 Pointer: 0x100b3b4e0
Operation: +

Value 0x100b3b4b0
Data: 0.456012
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x100b3b450
Child2 Pointer: 0x100b3b480
Operation: +

Value 0x100b3b450
Data: 0.153512
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x100b3b3d0
Child2 Pointer: 0x100b3b420
Operation: +

Value 0x100b3b3d0
Data: 0.000000
Grad: -13.583591
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b420
Data: 0.153512
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x965008ab0
Child2 Pointer: 0x100b3afd0
Operation: *

Value 0x965008ab0
Data: 0.310000
Grad: -6.726595
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3afd0
Data: 0.495200
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3af70
Child2 Pointer: 0x100b3afa0
Operation: +

Value 0x100b3af70
Data: 0.106400
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3af10
Child2 Pointer: 0x100b3af40
Operation: +

Value 0x100b3af10
Data: 0.056000
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3ae90
Child2 Pointer: 0x100b3aee0
Operation: +

Value 0x100b3ae90
Data: 0.000000
Grad: -11.292771
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3aee0
Data: 0.056000
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008900
Child2 Pointer: 0x100b3a560
Operation: *

Value 0x965008900
Data: 0.160000
Grad: -3.952470
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a560
Data: 0.350000
Grad: -1.806843
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3af40
Data: 0.050400
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008930
Child2 Pointer: 0x100b3a590
Operation: *

Value 0x965008930
Data: 0.240000
Grad: -2.371482
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a590
Data: 0.210000
Grad: -2.710265
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3afa0
Data: 0.388800
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008960
Child2 Pointer: 0x100b3a5c0
Operation: *

Value 0x965008960
Data: 0.720000
Grad: -6.098097
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a5c0
Data: 0.540000
Grad: -8.130795
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b480
Data: 0.302500
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x965008ae0
Child2 Pointer: 0x100b3b1b0
Operation: *

Value 0x965008ae0
Data: 0.660000
Grad: -6.225820
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b1b0
Data: 0.458334
Grad: -8.965171
isLeaf: 0
Child1 Pointer: 0x100b3afd0
Child2 Pointer: 0x0
Operation: t

Value 0x100b3afd0
Data: 0.495200
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3af70
Child2 Pointer: 0x100b3afa0
Operation: +

Value 0x100b3af70
Data: 0.106400
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3af10
Child2 Pointer: 0x100b3af40
Operation: +

Value 0x100b3af10
Data: 0.056000
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x100b3ae90
Child2 Pointer: 0x100b3aee0
Operation: +

Value 0x100b3ae90
Data: 0.000000
Grad: -11.292771
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3aee0
Data: 0.056000
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008900
Child2 Pointer: 0x100b3a560
Operation: *

Value 0x965008900
Data: 0.160000
Grad: -3.952470
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a560
Data: 0.350000
Grad: -1.806843
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3af40
Data: 0.050400
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008930
Child2 Pointer: 0x100b3a590
Operation: *

Value 0x965008930
Data: 0.240000
Grad: -2.371482
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a590
Data: 0.210000
Grad: -2.710265
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3afa0
Data: 0.388800
Grad: -11.292771
isLeaf: 0
Child1 Pointer: 0x965008960
Child2 Pointer: 0x100b3a5c0
Operation: *

Value 0x965008960
Data: 0.720000
Grad: -6.098097
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a5c0
Data: 0.540000
Grad: -8.130795
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b4e0
Data: 0.752192
Grad: -13.583591
isLeaf: 0
Child1 Pointer: 0x965008b10
Child2 Pointer: 0x100b3b360
Operation: *

Value 0x965008b10
Data: 0.460000
Grad: -22.211887
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b360
Data: 1.635200
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x100b3b300
Child2 Pointer: 0x100b3b330
Operation: +

Value 0x100b3b300
Data: 1.243200
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x100b3b2a0
Child2 Pointer: 0x100b3b2d0
Operation: +

Value 0x100b3b2a0
Data: 0.960000
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x100b3b040
Child2 Pointer: 0x100b3b270
Operation: +

Value 0x100b3b040
Data: 0.000000
Grad: -6.248452
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b270
Data: 0.960000
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x965008990
Child2 Pointer: 0x100b3a680
Operation: *

Value 0x965008990
Data: 0.960000
Grad: -6.248452
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a680
Data: 1.000000
Grad: -5.998514
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b2d0
Data: 0.283200
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x9650089c0
Child2 Pointer: 0x100b3a6b0
Operation: *

Value 0x9650089c0
Data: 0.590000
Grad: -2.999257
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a6b0
Data: 0.480000
Grad: -3.686587
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b330
Data: 0.392000
Grad: -6.248452
isLeaf: 0
Child1 Pointer: 0x9650089f0
Child2 Pointer: 0x100b3a6e0
Operation: *

Value 0x9650089f0
Data: 0.700000
Grad: -3.499133
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3a6e0
Data: 0.560000
Grad: -4.373917
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: 

Value 0x100b3b540
Data: -8.000000
Grad: -13.583591
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: """
graph = visualize_computational_graph(data_input)
graph.render('my_neural_net', format='png', view=True)