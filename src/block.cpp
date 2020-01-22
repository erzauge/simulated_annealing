#include "block.hpp"
#include "Logging.hpp"

block::block() : start(0), stop(0), step(0) {}

block::block(double start_, double step_, double stop_) {
  if (start_ > stop_) {
    LOG(LOG_ERROR) << "wrong order";
    return;
  }
  start = start_;
  stop = stop_;
  step = step_;
}

int block::size() const {
  if ((start + (int)((stop - start) / step) * step) < stop * 0.9999) {
    return (stop - start) / step + 2;
  }
  return (stop - start) / step + 1;
}

double block::operator[](int i) const {
  double r = start + i * step;
  if (r > stop) {
    return stop;
  }
  return r;
}

void block::operator()(double *b) {
  for (int i = 0; i < size(); ++i) {
    b[i] = operator[](i);
  }
}

std::ostream &operator<<(std::ostream &stream, const block &obj) {
  stream << obj.start << "\t" << obj.step << "\t" << obj.stop;
  return stream;
}

std::istream &operator>>(std::istream &stream, block &obj) {
  stream >> obj.start >> obj.step >> obj.stop;

  return stream;
}
