package absin.nn.scratch;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class NeuralNetwork {
	private INDArray input;
	private INDArray weights1;
	private INDArray weights2;
	private INDArray y;
	private INDArray output;
	private INDArray layer1;

	public NeuralNetwork(INDArray x, INDArray y) {
		super();
		this.input = x;
		this.weights1 = Nd4j.rand(new long[] { this.input.shape()[1], 9 });
		this.weights2 = Nd4j.rand(new long[] { 9, 1 });
		this.y = y;
		this.output = Nd4j.zeros(this.y.shape());
	}

	public static void main(String[] args) {
		INDArray x = Nd4j.create(new double[] { 0d, 0d, 1d, 0d, 1d, 1d, 1d, 0d, 1d, 1d, 1d, 1d }, new int[] { 4, 3 });
		INDArray y = Nd4j.create(new double[] { 0d, 1d, 1d, 0d }, new int[] { 4, 1 });
		System.out.println("Input-->" + y.transpose());
		NeuralNetwork nn = new NeuralNetwork(x, y);
		for (int i = 1; i <= 1500; i++) {
			nn.feedforward();
			nn.backprop();
			if (i % 100 == 0) {
				System.out.println("Pass-->" + i);
				nn.printLoss();
			}
		}
		System.out.println("Output-->" + nn.output.transpose());

	}

	private void printLoss() {
		INDArray loss = output.sub(y);
		System.out.println("\t Loss-->" + (loss.mul(loss).transpose()));
	}

	private static INDArray sigmoid(INDArray x) {
		INDArray sigmoid = Transforms.sigmoid(x);
		return sigmoid;
	}

	private static INDArray sigmoid_derivative(INDArray x) {
		// return x.mul(Nd4j.ones(x.shape()).sub(x));
		return Transforms.sigmoidDerivative(x, true);
	}

	private void feedforward() {
		this.layer1 = sigmoid(this.input.mmul(this.weights1));
		this.output = sigmoid(this.layer1.mmul(this.weights2));
	}

	private void backprop() {
		INDArray d_weights2 = layer1.transpose().mmul(y.sub(output).mul(2).mul(sigmoid_derivative(output)));
		INDArray d_weights1 = input.transpose()
				.mmul((y.sub(output).mul(2).mul(sigmoid_derivative(output)).mmul(weights2.transpose()))
						.mul(sigmoid_derivative(layer1)));
		this.weights1 = this.weights1.add(d_weights1);
		this.weights2 = this.weights2.add(d_weights2);
	}
}
