from typing import Any, Dict, List

import numpy as np


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[np.ndarray]
            The input values of the given node.

        Returns
        -------
        output: np.ndarray
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]

class SubOp(Op):
    """Op to element-wise sub two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise sub of input values."""
        assert len(input_values) == 2
        return input_values[0] - input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, mul_by_const(output_grad, -1.0)]
    
class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of input values."""
        """TODO: Your code here"""
        assert len(input_values) == 2
        try:
            r = input_values[0] * input_values[1] # TODO
        except ValueError as e:
            print("Caught ValueError:", e)
            print(node.name)
            raise
        return r
    
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        """TODO: Your code here"""
        return [mul(node.inputs[1], output_grad), mul(node.inputs[0], output_grad)]

class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of the input value and the constant."""
        """TODO: Your code here"""
        assert len(input_values) == 1
        return input_values[0] * node.constant
    
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        """TODO: Your code here"""
        return [mul_by_const(output_grad, node.constant)]

class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of input values."""
        """TODO: Your code here"""
        assert len(input_values) == 2
        return input_values[0] / input_values[1]
    
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        """TODO: Your code here"""
        return [div(output_grad, node.inputs[1]), 
                mul(div(node.inputs[0], mul_by_const(mul(node.inputs[1], node.inputs[1]), -1.0)), output_grad)]

class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of the input value and the constant."""
        """TODO: Your code here"""
        assert len(input_values) == 1
        return input_values[0] / node.constant
    
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        """TODO: Your code here"""
        return [div_by_const(output_grad, node.constant)]

class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(
        self, node_A: Node, node_B: Node, trans_A: bool = False, trans_B: bool = False
    ) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix
        trans_A: bool
            A boolean flag denoting whether to transpose A before multiplication.
        trans_B: bool
            A boolean flag denoting whether to transpose B before multiplication.

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"trans_A": trans_A, "trans_B": trans_B},
            name=f"({node_A.name + ('.T' if trans_A else '')}@{node_B.name + ('.T' if trans_B else '')})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the matrix multiplication result of input values.

        Note
        ----
        For this assignment, you can assume the matmul only works for 2d matrices.
        That being said, the test cases guarantee that input values are
        always 2d numpy.ndarray.
        """
        """TODO: Your code here"""
        assert len(input_values) == 2
        mat_A =  input_values[0].T if node.attrs["trans_A"] else input_values[0]
        mat_B =  input_values[1].T if node.attrs["trans_B"] else input_values[1]

        return mat_A @ mat_B

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input.

        Note
        ----
        - Same as the `compute` method, you can assume that the input are 2d matrices.
        However, it would be a good exercise to think about how to handle
        more general cases, i.e., when input can be either 1d vectors,
        2d matrices, or multi-dim tensors.
        - You may want to look up some materials for the gradients of matmul.
        """
        """TODO: Your code here"""
        A_T = node.trans_A
        B_T = node.trans_B
        if A_T:
            grad_1 = matmul(node.inputs[1], output_grad, trans_A=B_T, trans_B=True)
        else:
            grad_1 = matmul(output_grad, node.inputs[1], trans_B=not B_T)

        if B_T:
            grad_2 = matmul(output_grad, node.inputs[0], trans_A=True, trans_B=A_T)
        else:
            grad_2 = matmul(node.inputs[0], output_grad, trans_A=not A_T)
        return [grad_1, grad_2]

class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.zeros(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.ones(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class SumOp(Op): # TODO
    """Sum op that sum along a given axis. Only apply in 2d case """

    def __call__(self, node_A: Node, axis: int = 0, keepdims: bool = False) -> Node:
        return Node(
            inputs=[node_A], 
            op=self,
            attrs={"axis": axis, "keepdims": keepdims}, 
            name=f"Sum({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        assert len(input_values) == 1
        return np.sum(input_values[0], axis=node.axis, keepdims=node.keepdims)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [broadcast_to(output_grad, node.inputs[0])]
    
class BroadCastToOp(Op): # TODO
    """Braodcast op that broadcast A to B. Only apply in 2d case """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(inputs=[node_A, node_B], op=self, name=f"Broadcastto({node_A.name}, {node_B.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        assert len(input_values) == 2
        return np.broadcast_to(input_values[0], input_values[1].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [sum(output_grad, axis=0, keepdims=False), zeros_like(node.inputs[1])]

class ExpOp(Op):
    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"exp({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        return np.exp(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [mul(output_grad, exp(node.inputs[0]))]   
    
class LogOp(Op):
    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"log({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        return np.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [div(output_grad, node.inputs[0])]
    
# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
broadcast_to = BroadCastToOp()
sum = SumOp()
exp = ExpOp()
log = LogOp()
sub = SubOp()

def topo_sort(eval_nodes: List[Node], reversed: bool) -> List[Node]:
    """Given eval_nodes, return the topological order of the computation graph containing these nodes."""
    visited = set()
    topo_order = []
    
    def dfs(node : Node) -> None:
        visited.add(node)
        for next in node.inputs:
            if next not in visited:
                dfs(next)
        topo_order.append(node)
    
    for node in eval_nodes:
        dfs(node)
    if reversed:
        topo_order.reverse()
    return topo_order

class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, np.ndarray]) -> List[np.ndarray]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, np.ndarray]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[np.ndarray]
            The list of values for nodes in `eval_nodes` field.
        """
        """TODO: Your code here"""
        node_to_val = input_values.copy()
        sorted = topo_sort(self.eval_nodes, False)
        for node in sorted:
            if node in node_to_val:
                continue
            for input_node in node.inputs:
                if input_node not in node_to_val:
                    raise ValueError(f"Node {node} depends on {input_node}, but its value is missing.")
            inputs = [node_to_val[input_node] for input_node in node.inputs]
            node_to_val[node] = node.op.compute(node, inputs)
        
        eval_values = [node_to_val[node] for node in self.eval_nodes]
        return eval_values


def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """

    """TODO: Your code here"""
    topo_order = topo_sort([output_node], True)
    node_to_grad = {output_node : ones_like(output_node)}

    for node in topo_order:
        if not node.inputs:
            continue
        ouput_grad = node_to_grad[node]
        input_grads = node.op.gradient(node, ouput_grad)
        for input_n, input_g in zip(node.inputs, input_grads):
            if input_n in node_to_grad:
                node_to_grad[input_n] = add(node_to_grad[input_n], input_g)
            else:
                node_to_grad[input_n] = input_g

    grad_nodes = [node_to_grad[n] for n in nodes]
    return grad_nodes