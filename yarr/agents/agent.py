from abc import ABC, abstractmethod
from typing import Any, List


class Summary(object):
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value


class ScalarSummary(Summary):
    pass


class HistogramSummary(Summary):
    pass


class ImageSummary(Summary):
    pass


class TextSummary(Summary):
    pass


class VideoSummary(Summary):
    def __init__(self, name: str, value: Any, fps: int = 30):
        super(VideoSummary, self).__init__(name, value)
        self.fps = fps


class ActResult(object):

    def __init__(self, action: Any,
                 observation_elements: dict = None,
                 replay_elements: dict = None,
                 info: dict = None):
        self.action = action
        self.observation_elements = observation_elements or {}
        self.replay_elements = replay_elements or {}
        self.info = info or {}


class Agent(ABC):

    @abstractmethod
    def build(self, training: bool, device=None) -> None:
        pass

    @abstractmethod
    def update(self, step: int, replay_sample: dict) -> dict:
        pass

    @abstractmethod
    def act(self, step: int, observation: dict, deterministic: bool) -> ActResult:
        # returns dict of values that get put in the replay.
        # One of these must be 'action'.
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def update_summaries(self) -> List[Summary]:
        pass

    @abstractmethod
    def act_summaries(self) -> List[Summary]:
        pass

    @abstractmethod
    def load_weights(self, savedir: str) -> None:
        pass

    @abstractmethod
    def save_weights(self, savedir: str) -> None:
        pass


class BimanualAgent(Agent):
    """
    
    """

    def __init__(self, right_agent: Agent, left_agent: Agent):
        self.right_agent = right_agent
        self.left_agent = left_agent

    def build(self, training: bool, device=None) -> None:
        self.right_agent.build(training, device)
        self.left_agent.build(training, device)

    def update(self, step: int, replay_sample: dict) -> dict:
        raise Exception("not implemented")

    def act(self, step: int, observation: dict, deterministic: bool) -> ActResult:

        observation_elements = {}
        info = {}

        right_observation = {}
        left_observation = {}

        for k, v in observation.items(): 
            if "rgb" in k or "point_cloud" in k or "camera" in k:
                right_observation[k] = v
                left_observation[k] = v
            elif "right_" in k :
                right_observation[k[6:]] = v
            elif "left_" in k:
                left_observation[k[5:]] = v
            else:
                right_observation[k] = v
                left_observation[k] = v

        right_act_result = self.right_agent.act(step, right_observation, deterministic)
        left_act_result = self.left_agent.act(step, left_observation, deterministic)

        action = (*right_act_result.action, *left_act_result.action)

        observation_elements.update(right_act_result.observation_elements)
        observation_elements.update(left_act_result.observation_elements)

        info.update(right_act_result.info)
        info.update(left_act_result.info)

        return ActResult(action, observation_elements=observation_elements, info=info)

    def reset(self) -> None:
        self.right_agent.reset()
        self.left_agent.reset()

    def update_summaries(self) -> List[Summary]:
        return self.right_agent.update_summaries() + self.left_agent.update_summaries()

    def act_summaries(self) -> List[Summary]:
        return self.right_agent.act_summaries() + self.left_agent.act_summaries()
    
    def load_weights(self, savedir: str) -> None:
        self.right_agent.load_weights(savedir.replace("%ROBOT_NAME%", "right"))
        self.left_agent.load_weights(savedir.replace("%ROBOT_NAME%", "left"))

    def save_weights(self, savedir: str) -> None:
        raise Exception("not implemented")
