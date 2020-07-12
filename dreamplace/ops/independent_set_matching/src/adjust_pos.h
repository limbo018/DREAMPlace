/**
 * File              : adjust_pos.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.06.2020
 * Last Modified Date: 05.06.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_ADJUST_POS_H
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_ADJUST_POS_H

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
inline DREAMPLACE_HOST_DEVICE bool adjust_pos(T& x, T width, const Space<T>& space) {
  // the order is very tricky for numerical stability 
  x = DREAMPLACE_STD_NAMESPACE::min(x, space.xh - width);
  x = DREAMPLACE_STD_NAMESPACE::max(x, space.xl);
  return width + space.xl <= space.xh; 
}

DREAMPLACE_END_NAMESPACE

#endif
