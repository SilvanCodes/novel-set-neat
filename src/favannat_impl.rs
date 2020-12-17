use std::collections::HashMap;

use favannat::network::{EdgeLike, NetLike, NodeLike, Recurrent};

use crate::{
    genes::{
        activations,
        connections::{Connection, FeedForward},
        nodes::{Input, Node, Output},
        Activation, Id, Weight,
    },
    Individual,
};

impl NodeLike for Node {
    fn id(&self) -> usize {
        self.id().0
    }
    fn activation(&self) -> fn(f64) -> f64 {
        match self.1 {
            Activation::Linear => activations::LINEAR,
            Activation::Sigmoid => activations::SIGMOID,
            Activation::Gaussian => activations::GAUSSIAN,
            Activation::Tanh => activations::TANH,
            Activation::Step => activations::STEP,
            Activation::Sine => activations::SINE,
            Activation::Cosine => activations::COSINE,
            Activation::Inverse => activations::INVERSE,
            Activation::Absolute => activations::ABSOLUTE,
            Activation::Relu => activations::RELU,
            Activation::Squared => activations::SQUARED,
        }
    }
}

impl EdgeLike for Connection {
    fn start(&self) -> usize {
        self.input().0
    }
    fn end(&self) -> usize {
        self.output().0
    }
    fn weight(&self) -> f64 {
        (self.1).0
    }
}

impl NetLike<Node, Connection> for Individual {
    fn nodes(&self) -> Vec<&Node> {
        self.genome.nodes().collect()
    }
    fn edges(&self) -> Vec<&Connection> {
        self.feed_forward.as_sorted_vec()
    }
    fn inputs(&self) -> Vec<&Node> {
        self.inputs.as_sorted_vec()
    }
    fn outputs(&self) -> Vec<&Node> {
        self.outputs.as_sorted_vec()
    }
}

impl Recurrent<Node, Connection> for Individual {
    type Net = Self;

    fn unroll(&self) -> Self::Net {
        let mut unrolled_genome = self.clone();

        // maps recurrent connection input to wrapped actual input
        let mut unroll_map: HashMap<Id, Id> = HashMap::new();
        let mut tmp_ids = (0..usize::MAX).rev();

        for recurrent_connection in self.recurrent.as_sorted_vec() {
            let recurrent_input = unroll_map
                .entry(recurrent_connection.input())
                .or_insert_with(|| {
                    let wrapper_input_id = Id(tmp_ids.next().unwrap());

                    let wrapper_input_node = Input(Node(wrapper_input_id, Activation::Linear));
                    let wrapper_output_node =
                        Output(Node(Id(tmp_ids.next().unwrap()), Activation::Linear));

                    // used to carry value into next evaluation
                    let outward_wrapping_connection = FeedForward(Connection(
                        recurrent_connection.input(),
                        Weight(1.0),
                        Node::id(&*wrapper_output_node),
                    ));

                    // add nodes for wrapping
                    unrolled_genome.inputs.insert(wrapper_input_node);
                    unrolled_genome.outputs.insert(wrapper_output_node);

                    // add outward wrapping connection
                    unrolled_genome
                        .feed_forward
                        .insert(outward_wrapping_connection);

                    wrapper_input_id
                });

            let inward_wrapping_connection = FeedForward(Connection(
                *recurrent_input,
                recurrent_connection.1,
                recurrent_connection.output(),
            ));

            unrolled_genome
                .feed_forward
                .insert(inward_wrapping_connection);
        }
        unrolled_genome
    }

    fn recurrent_edges(&self) -> Vec<&Connection> {
        self.recurrent.as_sorted_vec()
    }
}

#[cfg(test)]
mod tests {
    use favannat::network::Recurrent;

    use crate::{Individual, Parameters};

    #[test]
    fn unroll_genome() {
        todo!("update to individual");
        /* let mut parameters: Parameters = Default::default();
        parameters.mutation.weights.perturbation_std_dev = 1.0;

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;
        parameters.mutation.recurrent = 1.0;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init(&mut context, &parameters);

        // should add recurrent connection from input to output
        assert!(genome_0.add_connection(&mut context, &parameters).is_ok());
        // dont add same connection twice
        assert!(genome_0.add_connection(&mut context, &parameters).is_err());

        assert_eq!(genome_0.recurrent.len(), 1);

        let genome_1 = genome_0.unroll();

        assert_eq!(genome_1.hidden.len(), 2);
        assert_eq!(genome_1.feed_forward.len(), 3); */
    }
}
