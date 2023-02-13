import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/execute', methods=['POST'])
def execute():
    user_input = request.form['code']
    main_code  = """
    import numpy as np
    isq2 = 1.0/(2.0**0.5)
    class Qstate:
      def __init__(self, n):
        self.n = n
        self.state = np.zeros(2**self.n, dtype=np.complex)
        self.state[0] = 1
      def op(self, t, i):
        eyeL = np.eye(2**i, dtype=np.complex)
        eyeR = np.eye(2**(self.n - i - int(t.shape[0]**0.5)), 
        dtype = np.complex)
        t_all = np.kron(np.kron(eyeL, t), eyeR)
        self.state = np.matmul(t_all, self.state)
      def hadamard(self, i):
        h_matrix = isq2 * np.array([
            [1,1],
            [1,-1]
        ])
        self.op(h_matrix, i)
      def t(self, i):
        t_matrix = np.array([
            [1,0],
            [0,isq2 + isq2 * 1j]
        ])
        self.op(t_matrix, i)
      def s(self, i):
        s_matrix = np.array([
            [1,0],
            [0,0+1j]
        ])
        self.op(s_matrix,i)
      def cnot(self, i):
        cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        self.op(cnot_matrix, i)
      def swap(self, i):
        swap_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        self.op(swap_matrix, i)   
    """
    input_code = """
    # calculate the probabilities of each bitstring
    probs = np.abs(q.state)**2

    # get the bitstrings and their corresponding probabilities
    result = [(np.binary_repr(i, q.n), p) for i, p in enumerate(probs)]

    # sort the result by probabilities in descending order
    result = sorted(result, key=lambda x: x[1], reverse=True)

    # output the bitstrings and their probabilities
    for bitstring, prob in result:
      print(f"Bitstring: {bitstring}, Probability: {prob}")

    print(q.state)
    """
    code = main_code + "\n" + user_input + "\n" + input_code
    exec(code)
    prediction = q.state
    return render_template('index.html', output='Predicted Weight in KGs :{}'.format(prediction))

if __name__ == '__main__':
	app.run(debug=True)
