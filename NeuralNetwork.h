
/*
	according to https://stackoverflow.com/questions/63139848/how-to-access-property-of-a-nested-struct-from-object-of-a-outer-struct,
	the structs in 'Layer' are not instatiated.
*/
struct Layer {
	int a=1;

	struct Convolutional {

	};

	struct Pooling: public Layer {
		// void forward(Layer) {

		// }
	};

	struct FullyConnected {

	};
};