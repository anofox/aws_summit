class Node(object):
    def __init__(self, key, value, parent):
        self.left = None
        self.right = None
        self.value = value
        self.key = key
        self.parent = parent

    def __str__(self):
        return ":".join(map(str, (self.key, self.value)))


class BinarySearchTree(object):
    def __init__(self):
        self.root = None

    def getRoot(self):
        return self.root

    def __setitem__(self, key, value):
        if(self.root == None):
            self.root = Node(key, value, None)
        else:
            self._set(key, value, self.root)

    def _set(self, key, value, node):
        if key == node.key:
            node.value = value
        elif key < node.key:
            if(node.left != None):
                self._set(key, value, node.left)
            else:
                node.left = Node(key, value, node)
        else:
            if(node.right != None):
                self._set(key, value, node.right)
            else:
                node.right = Node(key, value, node)

    def __contains__(self, key):
        return self.__getitem__(key) != None

    def __getitem__(self, key):
        if(self.root != None):
            return self._get(key, self.root)
        else:
            return None

    def _get(self, key, node):
        if key == node.key:
            return node.value
        elif key < node.key and node.left != None:
            return self._get(key, node.left)
        elif key > node.key and node.right != None:
            return self._get(key, node.right)


class FuzzySearchTree(BinarySearchTree):

    def _get(self, key, node):
        if key == node.key:
            return node.value
        elif key < node.key:
            if node.left != None:
                return self._get(key, node.left)
            else:
                return self._checkMin(key, node)
        else:
            if node.right != None:
                return self._get(key, node.right)
            else:
                return node.value # found the closest match that is larger

    def _checkMin(self, key, node):
        return node.value