import threading

import rclpy
from rclpy.node import Node

class GraspSelection(Node):
    def __init__(self):
        super().__init__("grasp_selection")

        ### parameters
        self.declare_parameter("pcd_topic", "/pointcloud")
        self.declare_parameter("candidates", 200)
        self.declare_parameter("top_k", 5)
        self.declare_parameter("voxel_size", 0.005)
        self.declare_parameter("seed", 0)
        self.declare_parameter("namespace", "camera_r")

        self.pcd_topic = self.get_parameter("pcd_topic").value
        self.candidates = self.get_parameter("candidates").value
        self.top_k = self.get_parameter("top_k").value
        self.voxel_size = self.get_parameter("voxel_size").value
        self.seed = self.get_parameter("seed").value
        self.namespace = self.get_parameter("namespace").value

        ### publishers
        self.setup_publishers()
        
        ### subscribers
        self.setup_subscribers()
        
        ### Get grasp candidates
        self.load_point_cloud()
        self.sample_grasps()

        ### publisher thread
        self.publisher_thread = threading.Thread(target=self.publish_grasps)
        self.publisher_thread.start()

    def setup_publishers(self):
        # publish a tf (in base_link frame) for each of the
        # top k grasp candidates
        pass

    def setup_subscribers(self):
        # all necessary subscriptions for the load_point_cloud function
        pass

    def load_point_cloud(self):
        # grab the point cloud from pcd_topic (a PointCloud2 message)
        # convert it into the base_link frame using the tf tree
        pass

    def sample_grasps(self):
        # run the sample grasps routinue
        # and save so we can publish!
        pass


    def publish_grasps(self):
        # publish the top k grasp candidates as transforms
        # in the base_link frame in separate topics
        pass



# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = GraspSelection()
        rclpy.spin(node)
    except Exception as exc:
        rclpy.logging.get_logger("grasp_selection").error(str(exc))
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    main()